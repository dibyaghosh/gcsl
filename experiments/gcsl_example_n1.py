#!/usr/bin/env python

import doodad as dd
import gcsl.doodad_utils as dd_utils
import argparse
import pdb
def run(output_dir='/tmp', env_name='pointmass_empty', gpu=True, seed=0, **kwargs):

    import gym
    import numpy as np
    from rlutil.logging import log_utils, logger

    import rlutil.torch as torch
    import rlutil.torch.pytorch_util as ptu

    # Envs

    from gcsl import envs
    from gcsl.envs.env_utils import DiscretizedActionEnv

    # Algo
    from gcsl.algo import buffer, gcsl_n1, variants_n1, networks_n1,gcsl_n1_sto

    ptu.set_gpu(gpu)
    if not gpu:
        print('Not using GPU. Will be slow.')

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = envs.create_env(env_name)
    env_params = envs.get_env_params(env_name)
    print(env_params)

    env, policy, replay_buffer, gcsl_kwargs = variants_n1.get_params(env, env_params)
    '''''
    algo = gcsl_n1.GCSL(
        env,
        policy,
        replay_buffer,
        **gcsl_kwargs
    )
    '''''
    algo = gcsl_n1_sto.GCSL(
        env,
        policy,
        replay_buffer,
        **gcsl_kwargs
    )

    exp_prefix = 'example/%s/gcsl_m1_sto/' % (env_name,)

    with log_utils.setup_logger(exp_prefix=exp_prefix, log_base_dir=output_dir):
        algo.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-S","--seed",default = '0' )
    parser.add_argument("-E","--env",default='pointmass_empty')

    args = parser.parse_args()
    seed = int(args.seed)
    env = args.env
    #pdb.set_trace()
    params = {
        'seed': [seed],
        'env_name': [env], #['lunar', 'pointmass_empty','pointmass_rooms', 'pusher', 'claw', 'door'],
        'gpu': [True],
    }
    dd_utils.launch(run, params, mode='local', instance_type='c4.xlarge')
