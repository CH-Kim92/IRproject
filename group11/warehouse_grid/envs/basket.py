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

    def _set_item(self, item):
        self.item = item
