import gym
from gym import spaces
import pygame
import numpy as np
from warehouse_grid.envs.map import map2D


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 300}

    def __init__(self):
        self.pygame = map2D()
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Dict(
            {
                "agent1": spaces.Box(np.array([0, 0, 0]), np.array([5, 5, 1]), dtype=int),
                "agent2": spaces.Box(np.array([0, 0, 0]), np.array([5, 5, 1]), dtype=int),

            }
        )
        print("ob_space", self.observation_space)

    def reset(self, seed=None, options=None):
        del self.pygame
        self.pygame = map2D()
        obs = self.pygame.start()
        info = {"distance": 5}
        return obs

    def step(self, action):
        # print(action)
        self.pygame.action(action)
        obs = tuple(self.pygame.observe())
        reward = self.pygame.evaluate()
        done = self.pygame.terminate()
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        self.pygame.view()
