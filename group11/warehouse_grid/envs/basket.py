import pygame
import math
import numpy as np


class basket:
    def __init__(self, status, pos, item=None):
        self.pos = pos
        self.status = status
        self.holding = 0
        self.item = item
        self.initpose = pos

    def _get_obs(self):
        return np.array(self.status, self.pos, self.item, dtype=int)

    # def _set_goal(self, goal):
    #     self.goal = goal

    def update(self, agent_location, action):
        for i in self.item:
            if i == agent_location and action == 5:
                self.item.delete(i)
        if self.item.isEmpty():
            self.status = 1

    def _set_item(self, item):
        self.item = item
