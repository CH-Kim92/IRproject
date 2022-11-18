import numpy as np
from .basket import basket
from .splitGoal import *


class test:
    def __init(self):
        self.basket1 = basket(0, np.array(
            [0, 5], dtype=int), np.array([[0, 0]], dtype=int))
        self.basket2 = basket(0, np.array(
            [1, 5], dtype=int), np.array([[3, 3]], dtype=int))
        self.basket3 = basket(0, np.array(
            [2, 5], dtype=int), np.array([[3, 2]], dtype=int))
        self.basket4 = basket(0, np.array(
            [3, 5], dtype=int), np.array([[0, 2]], dtype=int))
        self.basket5 = basket(0, np.array(
            [4, 5], dtype=int), np.array([[0, 4]], dtype=int))
