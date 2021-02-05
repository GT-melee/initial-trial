import math

import gym
from gym_minigrid.envs import EmptyEnv, MiniGridEnv, Grid, Goal
import numpy as np
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper


class _BobEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(self,
        size,
    ):
        self.size = size
        self.agent_start_pos = (1,1)
        self.agent_start_dir = 0

        super().__init__(
            grid_size=size,
            max_steps=4*size*size,
            # Set this to True for maximum speed
            see_through_walls=True
        )

        self.action_space = gym.spaces.Discrete(3)

    def step(self, action):
        obs, rew, done, info = super(_BobEnv, self).step(action)

        return obs, self.size * rew, done, info



    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        pos = np.random.randint(2,height-2+1,(2,)) if 2 < height - 2 else (3,3)

        self.put_obj(Goal(), pos[0], pos[1])

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

def BobEnv(size):
    return ImgObsWrapper(RGBImgPartialObsWrapper(_BobEnv(size)))

def GetBobEnvClass(size):
    def temp():
        return BobEnv(size)
    return temp