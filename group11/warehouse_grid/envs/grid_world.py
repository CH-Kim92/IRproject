import gym
from gym import spaces
import pygame
import numpy as np
from warehouse_grid.envs.map import map2D
import random


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(self):

        # Manually generate items in basket #
        # self.basket_items = np.array(
        #     [[[2, 4]], [[3, 1]], [[0, 3]], [[0, 2]], [[2, 4]]])

        # Randomly generate items in basket with replacement#
        # self.basket_items = np.random.randint(
        #     0, [5, 5], size=[5, 1, 2], dtype=int)

        # without replacement #
        sample_b = range(25)
        it = random.sample(sample_b, k=5)
        self.basket_items = np.array([[[int(it[0]/5), int(it[0] % 5)]], [[int(it[1]/5), int(it[1] % 5)]], [
                                     [int(it[2]/5), int(it[2] % 5)]], [[int(it[3]/5), int(it[3] % 5)]], [[int(it[4]/5), int(it[4] % 5)]]])
        print(self.basket_items)
        self.pygame = map2D(flag=0, ag1=None, ag2=None,
                            items=self.basket_items)
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "agent1": spaces.Box(np.array([0, 0]), np.array([6, 6]), dtype=int),
                "agent2": spaces.Box(np.array([0, 0]), np.array([6, 6]), dtype=int),
            }
        )

        self.action_sequence = None
        self.agents_location = None

    def reset(self, ag1, ag2, t1, t2, flag, items, seed=None, options=None):
        del self.pygame
        if flag == 1:
            self.pygame = map2D(flag=1, ag1=ag1, ag2=ag2,
                                items=items)
            self.set_targets(t1, t2)
            self.set_basket_items(items)
        else:
            self.pygame = map2D(flag=0, ag1=ag1, ag2=ag2,
                                items=items)
            self.set_targets(t1, t2)
            self.set_basket_items(items)

        obs = self.pygame.start()
        return obs

    def step(self, action):
        self.pygame.action(action)
        obs = self.pygame.observe()
        reward = self.pygame.evaluate()
        done = self.pygame.terminate()
        return obs, reward, done, {}

    def render(self, mode="human", close=False):
        self.pygame.sub_view()

    def final_render(self, mode='human', close=False):
        self.pygame.view()

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

    def get_basket_items(self):
        return self.basket_items

    def set_agents_location(self, agents_location):
        self.agents_location = agents_location

    def set_basket_items(self, items):
        self.pygame.set_basket_items(items)
