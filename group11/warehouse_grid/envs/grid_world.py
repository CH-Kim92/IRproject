import gym
from gym import spaces
import pygame
import numpy as np
from warehouse_grid.envs.map import map2D


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self):
        self.pygame = map2D(flag=0, ag1=None, ag2=None)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "agent1": spaces.Box(np.array([0, 0]), np.array([6, 6]), dtype=int),
                "agent2": spaces.Box(np.array([0, 0]), np.array([6, 6]), dtype=int),
            }
        )
        self.action_sequence = None
        self.agents_location = None

    def reset(self, ag1, ag2, t1, t2, seed=None, options=None):
        del self.pygame
        self.pygame = map2D(flag=1, ag1=ag1, ag2=ag2)
        self.set_targets(t1, t2)
        obs = self.pygame.start()
        return obs

    def step(self, action):
        # print(action)
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.terminate()
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        self.pygame.sub_view()

    def set_agents_location(self, agents):
        self.pygame.set_agents_location(agents)

    def set_targets(self, target1, target2):
        self.pygame.set_targets(target1, target2)

    def get_action_sequence(self):
        return self.pygame.get_action_sequence()

    def get_agents_location(self):
        return self.pygame.get_agents_location()

    def get_agents_initial_location(self):
        return self.pygame.get_agents_initial()

    def set_agents_location(self, agents_location):
        self.agents_location = agents_location
