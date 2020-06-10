import numpy as np
import gym

import rlutil.torch as torch
import rlutil.torch.distributions
import rlutil.torch.nn as nn
import torch.nn.functional as F
import rlutil.torch.pytorch_util as ptu
from torch.nn.parameter import Parameter

from gcsl import policy

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class FCNetwork(nn.Module):
    """
    A fully-connected network module
    """
    def __init__(self, dim_input, dim_output, layers=[256, 256],
            nonlinearity=torch.nn.ReLU, dropout=0):
        super(FCNetwork, self).__init__()
        net_layers = []
        dim = dim_input
        for i, layer_size in enumerate(layers):
          net_layers.append(torch.nn.Linear(dim, layer_size))
          net_layers.append(nonlinearity())
          if dropout > 0:
              net_layers.append(torch.nn.Dropout(0.4))
          dim = layer_size
        net_layers.append(torch.nn.Linear(dim, dim_output))
        self.layers = net_layers
        self.network = torch.nn.Sequential(*net_layers)

    def forward(self, states):
        return self.network(states)

class CBCNetwork(nn.Module):
    """
    A fully connected network which appends conditioning to each hidden layer
    """
    def __init__(self, dim_input, dim_conditioning, dim_output, layers=[256, 256],
            nonlinearity=torch.nn.ReLU, dropout=0, add_conditioning=True):
        super(CBCNetwork, self).__init__()
        
        self.dropout = bool(dropout != 0)
        self.add_conditioning = add_conditioning

        net_layers = torch.nn.ModuleList([])
        dim = dim_input + dim_conditioning
        
        for i, layer_size in enumerate(layers):
          net_layers.append(torch.nn.Linear(dim, layer_size))
          net_layers.append(nonlinearity())
          if self.dropout:
              net_layers.append(torch.nn.Dropout(dropout))
          if add_conditioning:
            dim = layer_size + dim_conditioning
          else:
            dim = layer_size

        net_layers.append(torch.nn.Linear(dim, dim_output))
        self.layers = net_layers

    def forward(self, states, conditioning):
        output = torch.cat((states, conditioning), dim=1)
        mod = 3 if self.dropout else 2
        for i in range(len(self.layers)):
            output = self.layers[i](output)
            if i % mod == mod - 1 and self.add_conditioning:
                output = torch.cat((output, conditioning), dim=1)
        return output
        
class MultiInputNetwork(nn.Module):
    def __init__(self, input_shapes, dim_out, input_embeddings=None, layers=[512, 512], freeze_embeddings=False):
        super(MultiInputNetwork, self).__init__()
        if input_embeddings is None:
            input_embeddings = [Flatten() for _ in range(len(input_shapes))]

        self.input_embeddings = input_embeddings
        self.freeze_embeddings = freeze_embeddings    
        
        dim_ins = [
            embedding(torch.tensor(np.zeros((1,) + input_shape))).size(1)
            for embedding, input_shape in zip(input_embeddings, input_shapes)
        ]
        
        full_dim_in = sum(dim_ins)
        self.net = FCNetwork(full_dim_in, dim_out, layers=layers)
    
    def forward(self, *args):
        assert len(args) == len(self.input_embeddings)
        embeddings = [embed_fn(x) for embed_fn,x in zip(self.input_embeddings, args)]
        embed = torch.cat(embeddings, dim=1)
        if self.freeze_embeddings:
            embed = embed.detach()
        return self.net(embed)

class StateGoalNetwork(nn.Module):
    def __init__(self, env, dim_out=1, state_embedding=None, goal_embedding=None, layers=[512, 512], max_horizon=None, freeze_embeddings=False, add_extra_conditioning=False, dropout=0):
        super(StateGoalNetwork, self).__init__()
        self.max_horizon = max_horizon
        if state_embedding is None:
            state_embedding = Flatten()
        if goal_embedding is None:
            goal_embedding = Flatten()
        
        self.state_embedding = state_embedding
        self.goal_embedding = goal_embedding
        self.freeze_embeddings = freeze_embeddings

        state_dim_in = self.state_embedding(torch.tensor(torch.zeros(env.observation_space.shape)[None])).size()[1]
        goal_dim_in = self.goal_embedding(torch.tensor(torch.zeros(env.goal_space.shape)[None])).size()[1]

        dim_in = state_dim_in + goal_dim_in

        if max_horizon is not None:
            self.net = CBCNetwork(dim_in, max_horizon, dim_out, layers=layers, add_conditioning=add_extra_conditioning, dropout=dropout)
        else:
            self.net = FCNetwork(dim_in, dim_out, layers=layers)

    def forward(self, state, goal, horizon=None):
        state = self.state_embedding(state)
        goal = self.goal_embedding(goal)
        embed = torch.cat((state, goal), dim=1)
        if self.freeze_embeddings:
            embed = embed.detach()

        if self.max_horizon is not None:
            horizon = self.process_horizon(horizon)
            output = self.net(embed, horizon)
        else:
            output = self.net(embed)
        return output
    
    def process_horizon(self, horizon):
        # Todo add format options
        return horizon

def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    batch_size, num_classes = logits.size()
    one_hot_mask = (torch.arange(0, num_classes)
                               .long()
                               .repeat(batch_size, 1)
                               .eq(target.data.repeat(num_classes, 1).t()))
    return logits.masked_select(one_hot_mask)

def cross_entropy_with_weights(logits, target, weights=None, label_smoothing=0):
    assert logits.dim() == 2
    assert not target.requires_grad
    target = target.squeeze(1) if target.dim() == 2 else target
    assert target.dim() == 1
    loss = torch.logsumexp(logits, dim=1) - (1-label_smoothing) * class_select(logits, target) - label_smoothing * logits.mean(dim=1)
    if weights is not None:
        # loss.size() = [N]. Assert weights has the same shape
        assert list(loss.size()) == list(weights.size())
        # Weight the loss
        loss = loss * weights
    return loss

class CrossEntropyLoss(nn.Module):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean', label_smoothing=0):
        super(CrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate
        self.label_smoothing = label_smoothing

    def forward(self, input, target, weights=None):
        ce = cross_entropy_with_weights(input, target, weights, self.label_smoothing)
        if self.aggregate == 'sum':
            return ce.sum()
        elif self.aggregate == 'mean':
            return ce.mean()
        elif self.aggregate is None:
            return ce

class DiscreteStochasticGoalPolicy(nn.Module, policy.GoalConditionedPolicy):
    def __init__(self, env, **kwargs):
        super(DiscreteStochasticGoalPolicy, self).__init__()
        
        self.action_space = env.action_space
        self.dim_out = env.action_space.n
        self.net = StateGoalNetwork(env, dim_out=self.dim_out, **kwargs)        

    def forward(self, obs, goal, horizon=None):
        return self.net.forward(obs, goal, horizon=horizon)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0,
            marginal_policy=None):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        
        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)
        
        logits = self.forward(obs, goal, horizon=horizon)
        if marginal_policy is not None:
            dummy_goal = torch.zeros_like(goal)
            marginal_logits = marginal_policy.forward(obs, dummy_goal, horizon)
            logits -= marginal_logits
        noisy_logits = logits  * (1 - noise)
        probs = torch.softmax(noisy_logits, 1)
        if greedy:
            samples = torch.argmax(probs, dim=-1)
        else:
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        return ptu.to_numpy(samples)
    
    def nll(self, obs, goal, actions, horizon=None):        
        logits = self.forward(obs, goal, horizon=horizon)
        return CrossEntropyLoss(aggregate=None, label_smoothing=0)(logits, actions, weights=None, )
    
    def probabilities(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        probs = torch.softmax(logits, 1)
        return probs

    def entropy(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        probs = torch.softmax(logits, 1)
        Z = torch.logsumexp(logits, dim=1)
        return Z - torch.sum(probs * logits, 1)

    def process_horizon(self, horizon):
        return horizon


class IndependentDiscretizedStochasticGoalPolicy(nn.Module, policy.GoalConditionedPolicy):
    def __init__(self, env, **kwargs):
        super(IndependentDiscretizedStochasticGoalPolicy, self).__init__()
        
        self.action_space = env.action_space
        self.n_dims = self.action_space.n_dims
        self.granularity = self.action_space.granularity
        dim_out = self.n_dims * self.granularity
        self.net = StateGoalNetwork(env, dim_out=dim_out, **kwargs)        

    def flattened(self, tensor):
        # tensor expected to be n x self.n_dims
        multipliers = self.granularity ** torch.tensor(np.arange(self.n_dims))
        flattened = (tensor * multipliers).sum(1)
        return flattened.int()
    
    def unflattened(self, tensor):
        # tensor expected to be n x 1
        digits = []
        output = tensor
        for _ in range(self.n_dims):
            digits.append(output % self.granularity)
            output = output // self.granularity
        uf = torch.stack(digits, dim=-1)
        return uf

    def forward(self, obs, goal, horizon=None):
        return self.net.forward(obs, goal, horizon=horizon)

    def act_vectorized(self, obs, goal, horizon=None, greedy=False, noise=0, marginal_policy=None):
        obs = torch.tensor(obs, dtype=torch.float32)
        goal = torch.tensor(goal, dtype=torch.float32)
        
        if horizon is not None:
            horizon = torch.tensor(horizon, dtype=torch.float32)
        
        logits = self.forward(obs, goal, horizon=horizon)
        logits = logits.view(-1, self.n_dims, self.granularity)
        noisy_logits = logits  * (1 - noise)
        probs = torch.softmax(noisy_logits, 2)

        if greedy:
            samples = torch.argmax(probs, dim=-1)
        else:
            samples = torch.distributions.categorical.Categorical(probs=probs).sample()
        samples = self.flattened(samples)
        if greedy:
            samples = ptu.to_numpy(samples)
            random_samples = np.random.choice(self.action_space.n, size=len(samples))
            return np.where(np.random.rand(len(samples)) < noise,
                    random_samples,
                    samples,
            )
        return ptu.to_numpy(samples)
    
    def nll(self, obs, goal, actions, horizon=None):        
        actions_perdim = self.unflattened(actions)
        # print(actions, self.flattened(actions_perdim))
        actions_perdim = actions_perdim.view(-1)

        logits = self.forward(obs, goal, horizon=horizon)
        logits_perdim = logits.view(-1, self.granularity)
        
        loss_perdim = CrossEntropyLoss(aggregate=None, label_smoothing=0)(logits_perdim, actions_perdim, weights=None)
        loss = loss_perdim.reshape(-1, self.n_dims)
        return loss.sum(1)
    
    def probabilities(self, obs, goal, horizon=None):
        """
        TODO(dibyaghosh): actually implement
        """
        raise NotImplementedError()

    def entropy(self, obs, goal, horizon=None):
        logits = self.forward(obs, goal, horizon=horizon)
        logits = logits.view(-1, self.n_dims, self.granularity)
        probs = torch.softmax(noisy_logits, 2)
        Z = torch.logsumexp(logits, dim=2)
        return (Z - torch.sum(probs * logits, 2)).sum(1)


