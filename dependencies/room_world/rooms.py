import numpy as np
import matplotlib.pyplot as plt
from multiworld.core.serializable import Serializable

from room_world.model_builder import MJCModel


def draw_wall(ax, start, end):
    if np.isclose(start[0],end[0]):
        ax.vlines(start[0], start[1], end[1], linewidth=4,)
    elif np.isclose(start[1], end[1]):
        ax.hlines(start[1], start[0], end[0], linewidth=4,)
    else:
        ax.plot(*np.array([start,end]).T, linewidth=4,c='black')

def draw_borders(ax, botLeft, topRight):
    botRight = [topRight[0], botLeft[1]]
    topLeft = [botLeft[0], topRight[1]]

    draw_wall(ax, botLeft, botRight)
    draw_wall(ax, botRight, topRight)
    draw_wall(ax, botLeft,  topLeft)
    draw_wall(ax, topLeft, topRight)

def draw_start_goal(ax, start, goal):
    ax.scatter([start[0]],[start[1]], c='r', s=400)
    ax.scatter([goal[0]],[goal[1]], c='g', s=400)


class RoomGenerator:
    def __init__(self, **kwargs):
        self.shell_config(**kwargs)

    @property
    def mjcmodel(self):
        return self._mjcmodel
    
    @property
    def worldbody(self):
        return self._worldbody
    
    def add_wall(self, start, end, name='wall', color="0.9 0.4 0.6 1"):
        raise NotImplementedError()
    
    def make_borders(self, botLeft, topRight, prefix='side', color="0.9 0.4 0.6 1"):
        botRight = [topRight[0], botLeft[1]]
        topLeft = [botLeft[0], topRight[1]]

        self.add_wall(botLeft, botRight, '%sS'%prefix, color)
        self.add_wall(botRight,topRight, '%sE'%prefix, color)
        self.add_wall(botLeft,  topLeft, '%sW'%prefix, color)
        self.add_wall(topLeft, topRight, '%sN'%prefix, color)

    def add_goal(self, position):
        self._worldbody.site(name="goal", pos=[*position, 0], size="0.01", rgba=[0,0.9,0.1,1])

    def shell_config(self, start_pos=(0,0), size=1):
        mjcmodel = MJCModel('blank')
        mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
        mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")
        default = mjcmodel.root.default()
        default.joint(damping=1, limited='false')
        default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

        worldbody = mjcmodel.root.worldbody()
        worldbody.camera(pos="0 0 %f"%(size*1.5), name="topview")
        self._mjcmodel = mjcmodel
        self._worldbody = worldbody

class PMRoomGenerator(RoomGenerator):
    def add_wall(self, start, end, name='wall', color="0.9 0.4 0.6 1"):
        self._worldbody.geom(conaffinity=1, fromto=[*start, .01, *end, .01], name=name, rgba=color, size=".03", type="capsule")

    def shell_config(self, start_pos=(0,0), size=1):
        mjcmodel = MJCModel('pointmass')
        mjcmodel.root.compiler(inertiafromgeom="true", angle="radian", coordinate="local")
        mjcmodel.root.option(timestep="0.01", gravity="0 0 0", iterations="20", integrator="Euler")

        # visual = mjcmodel.root.visual()
        # visual.headlight(ambient="0.5 0.5 0.5")

        default = mjcmodel.root.default()
        default.joint(damping=1, limited='false')
        default.geom(friction=".5 .1 .1", density="1000", margin="0.002", condim="1", contype="2", conaffinity="1")

        worldbody = mjcmodel.root.worldbody()
        worldbody.camera(pos="0 0 %f"%(size*1.5), name="topview")
        particle = worldbody.body(name='particle', pos=[*start_pos, 0])
        
        size_of_particle = 0.05

        particle.geom(name='particle_geom', type='sphere', size=size_of_particle, rgba='0 1.0 1.0 1', contype=1, mass=.01)
        particle.site(name='particle_site', pos=[0,0,0], size=0.01)
        particle.joint(name='ball_x', type='slide', pos=[0,0,0], axis=[1,0,0])
        particle.joint(name='ball_y', type='slide', pos=[0,0,0], axis=[0,1,0])

        actuator = mjcmodel.root.actuator()
        actuator.motor(joint="ball_x", ctrlrange=[-1.0, 1.0], gear=1, ctrllimited=True)
        actuator.motor(joint="ball_y", ctrlrange=[-1.0, 1.0], gear=1, ctrllimited=True)

        self._mjcmodel = mjcmodel
        self._worldbody = worldbody

class WheeledRoomGenerator(RoomGenerator):
    def add_wall(self, start, end, name='wall', color="0 0 0 0.3"):
        self._worldbody.geom(
            conaffinity=1,
            size=[max(abs(end[0]-start[0]) /2, 0.04) , max(abs(start[1]-end[1]) /2, 0.04),  0.4],
            pos=[(start[0]+end[0]) /2 , (start[1]+end[1]) /2 , 0.4],
            name=name,
            rgba=color,
            type="box"
        )

    def shell_config(self, start_pos=(0,0), size=8):
        mjcmodel = MJCModel('wheeled')
        mjcmodel.root.compiler(angle="radian",coordinate="local",inertiafromgeom="true",settotalmass="14")
        default = mjcmodel.root.default()
        default.geom(contype='1', conaffinity='0', condim='3', friction='.4 .4 .4', rgba='0.8 0.6 .4 1', solimp='0.0 0.8 0.01', solref='0.02 1')

        mjcmodel.root.size(nstack=300000, nuser_geom=1)
        mjcmodel.root.option(gravity="0 0 -9.81",timestep="0.01")
        asset = mjcmodel.root.asset()
        asset.texture(builtin="gradient",height="100",rgb1="1 1 1",rgb2="0 0 0",type="skybox",width="100")
        asset.texture(builtin="flat",height="1278",mark="cross",markrgb="1 1 1",name="texgeom",random="0.01",rgb1="0.8 0.6 0.4",rgb2="0.8 0.6 0.4",type="cube",width="127")
        asset.texture(builtin="checker",height="100",name="texplane",rgb1="1 1 1",rgb2="1 1 1",type="2d",width="100")
        asset.material(name="MatPlane",reflectance="0.5",shininess="0",specular="1",texrepeat="60 60",texture="texplane")
        asset.material(name="geom",texture="texgeom",texuniform="true")

        worldbody = mjcmodel.root.worldbody()
        worldbody.camera(pos="0 0 %f"%(size*1.5), name="topview")

        worldbody.light(cutoff="100",diffuse="1 1 1",dir="-0 0 -1.3",directional="true",exponent="1",pos="0 0 1.3",specular=".1 .1 .1")
        worldbody.geom(conaffinity="1",condim="3",material="MatPlane",name="floor",pos="0 0 0",rgba="0.8 0.9 0.8 1",size="40 40 40",type="plane")


        torso = worldbody.body(name="car",pos=[*start_pos, -0.4])
        
        torso.geom(name='body', type='box', pos ='0 0 .6', rgba="1 0 0 1", size='0.15 0.10 0.046')
        
        torso.joint(name="xmove", type="slide", limited="false", pos="0 0 0", axis="1 0 0", margin="0.01", armature="0", damping="0")
        torso.joint(name="ymove", type="slide", limited="false", pos="0 0 0", axis="0 1 0", margin="0.01", armature="0", damping="0")
        torso.joint(name="zmove", type="slide", limited="true", range="-1 0.03", pos="0 0 0", axis="0 0 1", margin="0.01", armature="0", damping="0")
        torso.joint(name="zrotate", type="hinge", limited="false", pos="0 0 0", axis="0 0 1", margin="0.01", armature="0", damping="0")

        wheelbody1 = torso.body(name="wheelbody1", pos="0 -0.2 0")
        wheelbody1.geom(name='wheel1', type='cylinder', axisangle="1 0 0 1.57", pos ='0 0 0.5', size='0.1 0.046', rgba='0 0 1 1')
        wheelbody1.joint(name="rotate_wheels1", type="hinge", limited="false", pos="0 0 0.5", axis="0 1 0", margin="0.01", armature="0", damping="0")

        wheelbody2 = torso.body(name="wheelbody2", pos="0 0.2 0")
        wheelbody2.geom(name='wheel2', type='cylinder', axisangle="1 0 0 1.57", pos ='0 0 0.5', size='0.1 0.046', rgba='0 0 1 1')
        wheelbody2.joint(name="rotate_wheels2", type="hinge", limited="false", pos="0 0 0.5", axis="0 1 0", margin="0.01", armature="0", damping="0")
        
        actuator = mjcmodel.root.actuator()
        actuator.velocity(name='rotate_wheels1', ctrlrange="-20 20", gear="1",   joint='rotate_wheels1')
        actuator.velocity(name='rotate_wheels2', ctrlrange="-20 20", gear="1" , joint='rotate_wheels2')

        self._mjcmodel = mjcmodel
        self._worldbody = worldbody

    def add_goal(self, position):
        self._worldbody.site(name="goal", pos=[*position, 0], size="0.05", rgba=[0,0.9,0.1,1])

class AntRoomGenerator(RoomGenerator): 
    def add_wall(self, start, end, name='wall', color="0.9 0.4 0.6 1"):
        self._worldbody.geom(
            conaffinity=1,
            size=[max(abs(end[0]-start[0]) /2, 0.1) , max(abs(start[1]-end[1]) /2, 0.1),  0.4],
            pos=[(start[0]+end[0]) /2 , (start[1]+end[1]) /2 , 0.4],
            name=name,
            rgba=color,
            type="box"
        )

    def shell_config(self, start_pos=(0,0), size=8):
        mjcmodel = MJCModel('ant_maze')
        mjcmodel.root.compiler(inertiafromgeom="true", angle="degree", coordinate="local")
        mjcmodel.root.option(timestep="0.01", gravity="0 0 -9.8", iterations="20", integrator="Euler")

        assets = mjcmodel.root.asset()
        assets.texture(builtin="gradient", height="100", rgb1="1 1 1", rgb2="0 0 0", type="skybox", width="100")
        assets.texture(builtin="flat", height="1278", mark="cross", markrgb="1 1 1", name="texgeom", random="0.01", rgb1="0.8 0.6 0.4", rgb2="0.8 0.6 0.4", type="cube", width="127")
        assets.texture(builtin="checker", height="100", name="texplane", rgb1="1 1 1", rgb2="1 1 1", type="2d", width="100")
        assets.material(name="MatPlane", reflectance="0.5", shininess="1", specular="1", texrepeat="60 60", texture="texplane")
        assets.material(name="geom", texture="texgeom", texuniform="true")

        default = mjcmodel.root.default()
        default.joint(armature="1", damping=1, limited='true')
        default.geom(friction="1 0.5 0.5", density="5.0", margin="0.01", condim="3", conaffinity="0")

        worldbody = mjcmodel.root.worldbody()
        worldbody.camera(pos="0 0 %f"%(size*1.5), name="topview")

        ant = worldbody.body(name='ant', pos=[*start_pos, 0.6])
        ant.geom(name='torso_geom', pos=[0, 0, 0], size="0.25", type="sphere")
        ant.joint(armature="0", damping="0", limited="false", margin="0.01", name="root", pos=[0, 0, 0], type="free")

        front_left_leg = ant.body(name="front_left_leg", pos=[0, 0, 0])
        front_left_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="aux_1_geom", size="0.08", type="capsule")
        aux_1 = front_left_leg.body(name="aux_1", pos=[0.2, 0.2, 0])
        aux_1.joint(axis=[0, 0, 1], name="hip_1", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
        aux_1.geom(fromto=[0.0, 0.0, 0.0, 0.2, 0.2, 0.0], name="left_leg_geom", size="0.08", type="capsule")
        ankle_1 = aux_1.body(pos=[0.2, 0.2, 0])
        ankle_1.joint(axis=[-1, 1, 0], name="ankle_1", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
        ankle_1.geom(fromto=[0.0, 0.0, 0.0, 0.4, 0.4, 0.0], name="left_ankle_geom", size="0.08", type="capsule")

        front_right_leg = ant.body(name="front_right_leg", pos=[0, 0, 0])
        front_right_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="aux_2_geom", size="0.08", type="capsule")
        aux_2 = front_right_leg.body(name="aux_2", pos=[-0.2, 0.2, 0])
        aux_2.joint(axis=[0, 0, 1], name="hip_2", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
        aux_2.geom(fromto=[0.0, 0.0, 0.0, -0.2, 0.2, 0.0], name="right_leg_geom", size="0.08", type="capsule")
        ankle_2 = aux_2.body(pos=[-0.2, 0.2, 0])
        ankle_2.joint(axis=[1, 1, 0], name="ankle_2", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
        ankle_2.geom(fromto=[0.0, 0.0, 0.0, -0.4, 0.4, 0.0], name="right_ankle_geom", size="0.08", type="capsule")

        back_left_leg = ant.body(name="back_left_leg", pos=[0, 0, 0])
        back_left_leg.geom(fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0], name="aux_3_geom", size="0.08", type="capsule")
        aux_3 = back_left_leg.body(name="aux_3", pos=[-0.2, -0.2, 0])
        aux_3.joint(axis=[0, 0, 1], name="hip_3", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
        aux_3.geom(fromto=[0.0, 0.0, 0.0, -0.2, -0.2, 0.0], name="backleft_leg_geom", size="0.08", type="capsule")
        ankle_3 = aux_3.body(pos=[-0.2, -0.2, 0])
        ankle_3.joint(axis=[-1, 1, 0], name="ankle_3", pos=[0.0, 0.0, 0.0], range=[-70, -30], type="hinge")
        ankle_3.geom(fromto=[0.0, 0.0, 0.0, -0.4, -0.4, 0.0], name="backleft_ankle_geom", size="0.08", type="capsule")

        back_right_leg = ant.body(name="back_right_leg", pos=[0, 0, 0])
        back_right_leg.geom(fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0], name="aux_4_geom", size="0.08", type="capsule")
        aux_4 = back_right_leg.body(name="aux_4", pos=[0.2, -0.2, 0])
        aux_4.joint(axis=[0, 0, 1], name="hip_4", pos=[0.0, 0.0, 0.0], range=[-30, 30], type="hinge")
        aux_4.geom(fromto=[0.0, 0.0, 0.0, 0.2, -0.2, 0.0], name="backright_leg_geom", size="0.08", type="capsule")
        ankle_4 = aux_4.body(pos=[0.2, -0.2, 0])
        ankle_4.joint(axis=[1, 1, 0], name="ankle_4", pos=[0.0, 0.0, 0.0], range=[30, 70], type="hinge")
        ankle_4.geom(fromto=[0.0, 0.0, 0.0, 0.4, -0.4, 0.0], name="backright_ankle_geom", size="0.08", type="capsule")

        worldbody.geom(conaffinity="1", condim="3", material="MatPlane", name="floor", pos=[0, 0, 0],
                    rgba="0.8 0.9 0.8 1", size="40 40 40", type="plane")

        actuator = mjcmodel.root.actuator()
        actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_4", gear="30")
        actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_4", gear="30")
        actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_1", gear="30")
        actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_1", gear="30")
        actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_2", gear="30")
        actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_2", gear="30")
        actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="hip_3", gear="30")
        actuator.motor(ctrllimited="true", ctrlrange="-1.0 1.0", joint="ankle_3", gear="30")

        self._mjcmodel = mjcmodel
        self._worldbody = worldbody

POSSIBLE_BASES = {
    'pm': PMRoomGenerator,
    'wheeled': WheeledRoomGenerator,
    'ant':  AntRoomGenerator
}

class Room(Serializable):
    
    def __init__(self, base='pm', length=1.2, width=1.2, start=None, target=None, starting_length=None, starting_width=None):
        super().quick_init(locals())

        assert base in POSSIBLE_BASES
        self.base = base

        self.length = length
        self.width = width

        if starting_length is None:
            starting_length = length
        if starting_width is None:
            starting_width = width
        
        self.starting_length = starting_length
        self.starting_width = starting_width

        if start is None:
            start = np.array((0,0))
        if target is None:
            target = np.array((0, 0))

        self.start = start
        self.target = target
         
        self.mjcmodel, self.worldbody = self.create_mjcmodel()
    
    def get_boundary(self):
        bottom_left = (-self.length / 2,- self.width / 2) 
        top_right = (self.length / 2 , self.width / 2)
        return bottom_left, top_right
    
    def get_starting_boundary(self):
        bottom_left = (-1 * self.starting_length / 2, -1 * self.starting_width / 2) 
        top_right = (self.starting_length / 2 , self.starting_width / 2)
        return bottom_left, top_right
    
    def get_walls(self):
        return []

    def create_mjcmodel(self):
    
        bl, tr = self.get_boundary()
        size = max(tr[0]-bl[0], tr[1]-bl[1])

        self.generator = POSSIBLE_BASES[self.base](start_pos=self.get_start(), size=size)        
        
        self.generator.make_borders(bl, tr)

        for n, (start,end) in enumerate(self.get_walls()):
            self.generator.add_wall(start, end , name='wall%d'%n)

        self.generator.add_goal(self.get_target())

        return self.generator.mjcmodel, self.generator.worldbody
    
    def get_mjcmodel(self):
        return self.mjcmodel

    def get_start(self):
        return self.start

    def get_target(self):
        return self.target
    
    def get_shaped_distance(self,a,b):
        return np.linalg.norm(a-b)

    def draw(self, ax=None, start=None, target=None):

        if ax is None:
            ax = plt.gca()
        
        if start is None:
            start = self.get_start()
        if target is None:
            target = self.get_target()
        
        bl, tr = self.get_boundary()
        draw_borders(ax, bl, tr)
        
        for wall_start, wall_end in self.get_walls():
            draw_wall(ax, wall_start, wall_end)

        draw_start_goal(ax, start, target)
        
        
    def XY(self, n=50):
        bl, tr = self.get_starting_boundary()
        X = np.linspace(bl[0] + 0.04, tr[0] - 0.04, n)
        Y = np.linspace(bl[1] + 0.04, tr[1] - 0.04, n)
        
        X,Y = np.meshgrid(X,Y)
        states = np.array([X.flatten(), Y.flatten()]).T
        return states

    def XYmesh(self, n=50):
        bl, tr = self.get_starting_boundary()
        X = np.linspace(bl[0] + 0.04, tr[0] - 0.04, n)
        Y = np.linspace(bl[1] + 0.04, tr[1] - 0.04, n)
        
        X,Y = np.meshgrid(X,Y)
        return X,Y


    def draw_reward(self, reward=None,ax=None):
        if ax is None:
            ax = plt.gca()
            
        if reward is None:
            reward = lambda x,y: -1 * self.get_shaped_distance(np.array([x,y]),self.get_target())
        
        X,Y = self.XYmesh()
        H = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                H[i,j] = reward(X[i,j],Y[i,j])
        return ax.contourf(X,Y,H,50)
    
    def draw_embedding(self, embedding=None, ax=None):
        raise NotImplementedError()


class RoomWithWall(Room, Serializable):
    
    def __init__(self, base='pm', length=1.2, width=1.2, start=None, target=None):
        Serializable.quick_init(self, locals())
        
        if start is None:
            start = np.array((-length / 6, -width / 4))
        
        if target is None:
            target = np.array(( - width / 6, width / 4))

        super().__init__(base, length, width, start, target)

    def get_boundary(self):
        bottom_left = (-self.length / 2, - self.width / 2)
        top_right = (self.length / 2 , self.width / 2)
        return bottom_left, top_right
    
    def get_starting_boundary(self):
        return self.get_boundary()

    def get_walls(self):
        return [
            [(-self.length / 2, 0), (self.length / 6, 0)],
        ]
    
    def get_shaped_distance(self,a,b):
        if a[1] * b[1] > 0 and abs(a[1]) > 0.05 and abs(b[1]) > 0.05:
            return np.linalg.norm(a-b)
        
        if a[1] < 0:
            a,b = b,a # make a on top, b on bottom
        
        intersection_point = a[0] + (b[0] - a[0]) * (a[1] / (a[1] - b[1]))
        if intersection_point > self.length / 6:
            return np.linalg.norm(a-b)
        
        midpoint = np.array([self.length / 6, 0])
        return np.linalg.norm(a - midpoint) + np.linalg.norm(midpoint - b)

class FourRoom(Room, Serializable):
    def __init__(self, base='pm', length=1.2, width=1.2, start=None, target=None):
        Serializable.quick_init(self, locals())

        width = length

        if start is None:
            start = np.array((-length/2 + 0.2, -width / 2 + 0.2))
        
        if target is None:
            target = np.array((length / 2 - 0.2, width / 2 - 0.2))

        super().__init__(base, length, width, start, target)

    def get_boundary(self):
        bottom_left = (-self.length / 2, - self.width / 2)
        top_right = (self.length / 2 , self.width / 2)
        return bottom_left, top_right
        
    def get_starting_boundary(self):
        return self.get_boundary()

    def get_walls(self):
        return [
            [(-self.length / 2, 0), (-self.length / 2 + self.length/6, 0)],
            [(-self.length / 2 + 2*self.length/6, 0), (-self.length / 2 + 4*self.length/6, 0)],
            [(-self.length / 2 + 5*self.length/6, 0), (-self.length / 2 + 6*self.length/6, 0)],
            [(0,-self.length / 2), (0,-self.length / 2 + self.length/6)],
            [(0, -self.length / 2 + 2*self.length/6), (0,-self.length / 2 + 4*self.length/6)],
            [(0, -self.length / 2 + 5*self.length/6), (0, -self.length / 2 + 6*self.length/6)],
        ]


    def get_shaped_distance(self,a,b):
        if a[0] * b[0] > 0 and a[1] *b[1] > 0:
            return np.linalg.norm(a-b)

        dist =  (self.length / 4) * (2)**0.5
        door_positions = np.array([
            [0, self.length /4],
            [self.length / 4, 0],
            [0, -self.length /4],
            [-self.length / 4, 0]
        ])
        
        precomputed_distances = np.array([
            [0, dist, 2 * dist, dist],
            [dist, 0, dist, 2 * dist],
            [2 * dist, dist, 0, dist],
            [dist, 2 * dist, dist, 0]
        ])

        def distance_thru_doors(a,b,door1,door2):
            da = np.linalg.norm(a-door_positions[door1])
            db = np.linalg.norm(b-door_positions[door2])
            dmid = precomputed_distances[door1,door2]
            return da + db + dmid

        doors = [[(2,3), (0,3)], [(1,2), (0,1)]]
        possible_start_doors = doors[a[0] > 0][a[1] > 0]
        possible_end_doors = doors[b[0] > 0][b[1]  > 0]

        dist = min([distance_thru_doors(a,b,door1, door2) for door1 in possible_start_doors for door2 in possible_end_doors])
        return dist
