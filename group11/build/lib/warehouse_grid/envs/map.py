import pygame
import math
import numpy as np
from .agent import agent
from .basket import basket
import random

Window_size = 500
Size = 5
pix_square_size = (
    Window_size / Size
)

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
GRAY = (128, 128, 128)


class map2D:
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 1}

    def __init__(self):
        self.render_mode = "human"
        random.seed(10)
        self.window = None
        self.canvas = pygame.Surface((Window_size, Window_size+100))
        self.clock = pygame.time.Clock()
        self.number_items = 25
        self.storage = []
        self.colours = []
        self.height = pix_square_size
        self.poss = []
        for i in range(self.number_items):
            self.colours.append((random.randint(
                0, 255), (random.randint(0, 255)), (random.randint(0, 255))))
        k = 0
        for r in range(Size):
            for c in range(Size):
                item_position = np.array([r, c], dtype=int)
                self.poss.append(item_position)
                self.storage.append([item_position, self.colours[k]])
                k += 1

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

        self.baskets = np.array(
            [self.basket1, self.basket2, self.basket3, self.basket4, self.basket5])

        self.items = []
        for i in self.baskets:
            for ii in i.item:
                self.items.append(ii)

        self.robot1 = agent((0, 0, 0), [0, 0])
        self.robot2 = agent((0, 50, 50), [3, 0])

    def action(self, action):
        if action == 0:  # right
            direction = np.array([1, 0], dtype=int)
        if action == 1:  # down
            direction = np.array([0, 1], dtype=int)
        if action == 2:  # left
            direction = np.array([-1, 0], dtype=int)
        if action == 3:  # up
            direction = np.array([0, -1], dtype=int)
        ## pick up ##
        if action == 4:
            self.robot.colour = self.get_colour(self.robot.pos)
        ## drop ##
        elif action == 5:
            self.robot.colour = self.robot.initcolour

        if action < 4:
            self.robot.pos = np.clip(self.robot.pos + direction, 0, Size - 1)
        # print(self.robot.pos)
        self.robot.update()

    def terminate(self):
        if self.robot.holding == 0:
            return False
        elif self.robot.holding == 1 and np.array_equal(self.robot.pos, self.basket2.item[0]):
            return True
        else:
            return False

    def evaluate(self):
        reward = 0
        # reach to the storage
        if np.array_equal(self.robot.pos, self.basket2.item[0]):
            reward = 100

        return reward

    # Agent observation space : position, holding
    def observe(self):
        position = self.robot._get_obs()
        holding = self.robot.holding
        observation = []
        for i in position:
            observation.append(i)
        observation.append(holding)
        return np.array(observation)

    def start(self):
        position = self.robot.initpose
        print("in start")
        print(position)
        holding = 0
        observation = []
        for i in position:
            observation.append(i)
        observation.append(holding)
        return tuple(observation)

    def get_colour(self, pos):
        for i in self.storage:
            if np.array_equal(i[0], pos):
                c = i[1]
        return c

    def view(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (Window_size, Window_size+100))
        self.canvas = pygame.Surface((Window_size, Window_size+100))
        self.canvas.fill((255, 255, 255))

        ### Drawing storages ###
        for i in self.storage:
            for c in self.items:
                if np.array_equal(i[0], c):
                    pygame.draw.rect(
                        self.canvas,
                        i[1],
                        pygame.Rect(
                            pix_square_size * i[0],
                            (pix_square_size-3, pix_square_size-3)
                        )
                    )

        ### Drawing grd ###

        ### Drawing robot ###
        # print(self.robot.colour)
        pygame.draw.circle(
            self.canvas,
            self.robot.colour,
            (np.array(self.robot.pos, dtype=int) + 0.5) * pix_square_size,
            pix_square_size / 5,
        )
        pygame.draw.circle(
            self.canvas,
            (0, 0, 0),
            (np.array(self.robot.pos, dtype=int) + 0.5) * pix_square_size,
            pix_square_size / 3,
            width=3
        )

        ### Drawing target ###
        # pygame.draw.rect(
        #     self.canvas,
        #     RED,
        #     pygame.Rect(pix_square_size*self.basket1.pos,
        #                 (pix_square_size, pix_square_size))
        # )

        for b in self.baskets:
            for i in range(len(b.item)):
                for k in range(len(self.poss)):
                    if np.array_equal(self.poss[k], b.item[i]):
                        pygame.draw.rect(
                            self.canvas,
                            self.storage[k][1],
                            pygame.Rect(
                                pix_square_size * b.pos+i *
                                pix_square_size/len(b.item),
                                (pix_square_size/len(b.item), pix_square_size)
                            )
                        )

        ### Draw Line ####
        for x in range(Size + 2):
            pygame.draw.line(
                self.canvas,
                0,
                (0, pix_square_size * x),
                (Window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                self.canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, Window_size),
                width=3,
            )

        if self.render_mode == "human":

            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
