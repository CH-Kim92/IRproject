import pygame
import math
import numpy as np


class agent:
    def __init__(self, pos=None, goal=None, holding_item=[]):
        self.goal = goal
        self.pos = pos
        self.holding = holding_item
        self.initpose = pos

    def _get_obs(self):
        position = self.pos
        obs = np.append(position, self.holding)
        return obs

    def _get_goals(self):
        return self.goal

    def _set_position(self, position):
        self.pos = position

    def _set_initpos(self):
        self.pos = self.initpose

    def _set_goal(self, goal):
        self.goal = goal

    def _set_holding_item(self, item):
        self.holding = item

    # def update(self):
    #     if self.initcolour == self.colour:
    #         self.holding = 0
    #     else:
    #         self.holding = 1
