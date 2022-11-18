import pygame
import math
import numpy as np


class agent:
    def __init__(self, colour, pos, goal, holding=0):
        self.goal = goal
        self.initcolour = colour
        self.colour = colour
        self.pos = pos
        self.holding = holding
        self.initpose = pos

    def _get_obs(self):
        position = self.pos
        obs = np.append(position, self.holding)
        return obs

    def _set_goal(self, goal):
        self.goal = goal

    def update(self):
        if self.initcolour == self.colour:
            self.holding = 0
        else:
            self.holding = 1
