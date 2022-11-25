import pygame
import math
import numpy as np
from .agent import agent
from .basket import basket
from .splitGoal import *
from .path import *
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
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, flag, ag1, ag2):
        self.render_mode = "rgb_array"
        random.seed(10)
        self.window = None
        self.canvas = pygame.Surface((Window_size, Window_size+100))
        self.clock = pygame.time.Clock()
        self.number_items = 25
        self.init = [[0, 0], [3, 0]]
        self.storage = []
        self.colours = []
        self.height = pix_square_size
        self.poss = []
        self.targets = []
        self.basket_items = []
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
            [0, 5], dtype=int), np.array([[0, 3]], dtype=int))
        self.basket2 = basket(0, np.array(
            [1, 5], dtype=int), np.array([[0, 3]], dtype=int))
        self.basket3 = basket(0, np.array(
            [2, 5], dtype=int), np.array([[3, 0]], dtype=int))
        self.basket4 = basket(0, np.array(
            [3, 5], dtype=int), np.array([[2, 2]], dtype=int))
        self.basket5 = basket(0, np.array(
            [4, 5], dtype=int), np.array([[2, 4]], dtype=int))

        self.baskets = np.array(
            [self.basket1, self.basket2, self.basket3, self.basket4, self.basket5])

        self.items = []
        for i in self.baskets:
            for ii in i.item:
                self.items.append(ii)
        self.robot1 = agent()
        self.robot2 = agent()
        if flag == 0:
            self.agents_location = [[0, 0], [3, 0]]
            self.sp_g = split_goal(self.agents_location, self.baskets)
            self.robot1._set_position(self.agents_location[0])
            self.robot2._set_position(self.agents_location[1])
            self.robot1._set_goal(self.sp_g[0])
            self.robot2._set_goal(self.sp_g[1])
            self.robot1_actions = path_planning(
                self.robot1.pos, self.robot1.goal)
            self.robot2_actions = path_planning(
                self.robot2.pos, self.robot2.goal)
        if flag == 1:
            self.robot1._set_position(ag1)
            self.robot2._set_position(ag2)
        self.robot1_done = False
        self.robot2_done = False

    def action(self, action):

        action1 = action[0]
        action2 = action[1]

        if action1 == 0:  # right
            direction1 = np.array([1, 0], dtype=int)
        elif action1 == 1:  # down
            direction1 = np.array([0, 1], dtype=int)
        elif action1 == 2:  # left
            direction1 = np.array([-1, 0], dtype=int)
        elif action1 == 3:  # up
            direction1 = np.array([0, -1], dtype=int)
        elif action1 == 4:  # waiting
            direction1 = np.array([0, 0], dtype=int)
        elif action1 == 5:  # waiting
            direction1 = np.array([0, 0], dtype=int)
        elif action1 == 6:  # waiting
            direction1 = np.array([0, 0], dtype=int)

        x_pos1 = np.clip(self.robot1.pos[0] + direction1[0], 0, 4)
        y_pos1 = np.clip(self.robot1.pos[1] + direction1[1], 0, 5)
        self.robot1.pos = np.array([x_pos1, y_pos1])

        if action2 == 0:  # right
            direction2 = np.array([1, 0], dtype=int)
        elif action2 == 1:  # down
            direction2 = np.array([0, 1], dtype=int)
        elif action2 == 2:  # left
            direction2 = np.array([-1, 0], dtype=int)
        elif action2 == 3:  # up
            direction2 = np.array([0, -1], dtype=int)
        elif action2 == 4:  # waiting
            direction2 = np.array([0, 0], dtype=int)
        elif action2 == 5:  # waiting
            direction2 = np.array([0, 0], dtype=int)
        elif action2 == 6:  # waiting
            direction2 = np.array([0, 0], dtype=int)
        # elif action2 == 4:
        #     self.robot2._set_holding_item(self.robot2.pos)
        # if action2 < 4:
        x_pos2 = np.clip(self.robot2.pos[0] + direction2[0], 0, 4)
        y_pos2 = np.clip(self.robot2.pos[1] + direction2[1], 0, 5)
        self.robot2.pos = np.array([x_pos2, y_pos2])

        # self.robot1.pos = self.robot1.pos + direction1
        # self.robot2.pos = self.robot2.pos + direction2

    def terminate(self):
        isdone = False
        # collision
        if np.array_equal(self.robot1.pos, self.robot2.pos):
            isdone = True

        # reach goals
        if np.array_equal(self.robot1.pos, self.targets[0]) and np.array_equal(self.robot2.pos, self.targets[1]):
            isdone = True
        return isdone

    ########### Reward Function #######
    def evaluate(self):
        reward = -10
        if np.array_equal(self.robot1.pos, self.robot2.pos):
            # print("***************collision*****************")
            reward = -100
        elif np.array_equal(self.robot1.pos, self.targets[0]) and np.array_equal(self.robot2.pos, self.targets[1]):
            # print("***************all done *****************")
            reward = 1000
        elif (np.array_equal(self.robot1.pos, self.targets[0])) and (not self.robot1_done):

            self.robot1_done = True
            # print("*******************robot 1*******************")
            reward = 500
        elif (np.array_equal(self.robot2.pos, self.targets[1])) and (not self.robot2_done):
            self.robot2_done = True
            # print("*******************robot 2******************* ")
            reward = 500
        elif (np.array_equal(self.robot2.pos, self.targets[0])) and (not self.robot1_done):
            self.robot1_done = True
            # print("*******************robot 2 in target 1*******************")
            reward = -50
        elif (np.array_equal(self.robot1.pos, self.targets[1])) and (not self.robot1_done):
            self.robot1_done = True
            # print("*******************robot 1 in target 2*******************")
            reward = -50

        return reward

    # Agent observation space : position, holding

    def _get_obs(self):
        return {"agent1": self.robot1.pos, "agent2": self.robot2.pos}

    def observe(self):
        position1 = self.robot1._get_obs()
        position2 = self.robot2._get_obs()
        observation = {'agent1': position1, 'agent2': position2}
        return observation

    def start(self):
        self.robot1.pos
        self.robot2.pos
        obs = self._get_obs()
        return obs

    def get_colour(self, pos):
        for i in self.storage:
            if np.array_equal(i[0], pos):
                c = i[1]
        return c

    def get_agents_location(self):
        return self.robot1.pos, self.robot2.pos

    def get_goal_list(self):
        return self.sp_g

    def get_agents_initial(self):
        return self.init

    def get_action_sequence(self):
        return self.robot1_actions, self.robot2_actions

    def set_agents_location(self, agents):
        self.robot1.pos = agents[0]
        self.robot2.pos = agents[1]

    def set_targets(self, target1, target2):
        self.targets.append(target1)
        self.targets.append(target2)

    def set_basketitmes(self, items):
        self.basket_items = items

    def sub_view(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (Window_size, Window_size+100))
        self.canvas = pygame.Surface((Window_size, Window_size+100))
        self.canvas.fill((255, 255, 255))

        ### Drawing robot ###
        # print(self.robot.colour)
        ## Drawing target ###
        pygame.draw.rect(
            self.canvas,
            RED,
            pygame.Rect(pix_square_size*self.targets[0],
                        (pix_square_size, pix_square_size))
        )
        pygame.draw.rect(
            self.canvas,
            BLUE,
            pygame.Rect(pix_square_size*self.targets[1],
                        (pix_square_size, pix_square_size))
        )
        pygame.draw.circle(
            self.canvas,
            RED,
            (np.array(self.robot1.pos, dtype=int) + 0.5) * pix_square_size,
            pix_square_size / 5,
        )
        pygame.draw.circle(
            self.canvas,
            BLUE,
            (np.array(self.robot2.pos, dtype=int) + 0.5) * pix_square_size,
            pix_square_size / 3,
            width=3
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

    # def view(self):
    #     if self.window is None and self.render_mode == "human":
    #         pygame.init()
    #         pygame.display.init()
    #         self.window = pygame.display.set_mode(
    #             (Window_size, Window_size+100))
    #     self.canvas = pygame.Surface((Window_size, Window_size+100))
    #     self.canvas.fill((255, 255, 255))

    #     ### Drawing storages ###
    #     for i in self.storage:
    #         for c in self.items:
    #             if np.array_equal(i[0], c):
    #                 pygame.draw.rect(
    #                     self.canvas,
    #                     i[1],
    #                     pygame.Rect(
    #                         pix_square_size * i[0],
    #                         (pix_square_size-3, pix_square_size-3)
    #                     )
    #                 )

    #     ### Drawing grd ###

    #     ### Drawing robot ###
    #     # print(self.robot.colour)
    #     pygame.draw.circle(
    #         self.canvas,
    #         self.robot.colour,
    #         (np.array(self.robot.pos, dtype=int) + 0.5) * pix_square_size,
    #         pix_square_size / 5,
    #     )
    #     pygame.draw.circle(
    #         self.canvas,
    #         (0, 0, 0),
    #         (np.array(self.robot.pos, dtype=int) + 0.5) * pix_square_size,
    #         pix_square_size / 3,
    #         width=3
    #     )

    #     ### Drawing target ###
    #     # pygame.draw.rect(
    #     #     self.canvas,
    #     #     RED,
    #     #     pygame.Rect(pix_square_size*self.basket1.pos,
    #     #                 (pix_square_size, pix_square_size))
    #     # )

    #     for b in self.baskets:
    #         for i in range(len(b.item)):
    #             for k in range(len(self.poss)):
    #                 if np.array_equal(self.poss[k], b.item[i]):
    #                     pygame.draw.rect(
    #                         self.canvas,
    #                         self.storage[k][1],
    #                         pygame.Rect(
    #                             pix_square_size * b.pos+i *
    #                             pix_square_size/len(b.item),
    #                             (pix_square_size/len(b.item), pix_square_size)
    #                         )
    #                     )

    #     ### Draw Line ####
    #     for x in range(Size + 2):
    #         pygame.draw.line(
    #             self.canvas,
    #             0,
    #             (0, pix_square_size * x),
    #             (Window_size, pix_square_size * x),
    #             width=3,
    #         )
    #         pygame.draw.line(
    #             self.canvas,
    #             0,
    #             (pix_square_size * x, 0),
    #             (pix_square_size * x, Window_size),
    #             width=3,
    #         )

    #     if self.render_mode == "human":

    #         # The following line copies our drawings from `canvas` to the visible window
    #         self.window.blit(self.canvas, self.canvas.get_rect())
    #         pygame.event.pump()
    #         pygame.display.update()

    #         # We need to ensure that human-rendering occurs at the predefined framerate.
    #         # The following line will automatically add a delay to keep the framerate stable.
    #         self.clock.tick(self.metadata["render_fps"])
