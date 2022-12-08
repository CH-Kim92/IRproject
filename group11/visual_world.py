from gym import spaces, Env
import pygame
import numpy as np
import random


class VisWorldEnv(Env):

    metadata = {"render_modes": "human", "render_fps": 5}

    def __init__(self, render_mode="human", size=5, debug=False, seed=1):
        self.size = size  # The size of the square grid
        self.window_size = 500  # The size of the PyGame window
        self.gridsize = 5
        self.step_nb = 0
        self.nb_steps = 1000
        self.b_status = [0, 0, 0, 0, 0]
        self.debug = debug
        self.render_mode = render_mode

        self.observation_space = spaces.MultiDiscrete([5, 5]*7)
        self.action_space = spaces.Discrete(49)

        np.random.seed(seed)

        self.state_agent = np.array([[0, 0], [4, 0]])
        self.a_holding = np.array([0, 0])
        self.state_target = np.random.randint(4, size=(5, 2))
        # self.state_agent = np.array([[0,0],[4,0]])
        # self.state_target = np.array([[0,2],[1,2],[2,2],[3,2],[4,2]])
        self.items = self.state_target.tolist()

        # self.render_mode = "human"
        # self.render_mode = "human"
        self.window = None
        self.clock = None

        self.action_map = {}
        for i in range(self.action_space.n):
            # Change i to base 5 (possible actions)
            action = [0] * 2  # for 2 agents
            num = i
            index = -1
            while num > 0:
                action[index] = num % 7  # number of actions 5
                num = num // 7  # number of actions 5
                index -= 1
            self.action_map[i] = action

    def _get_obs(self):
        return np.concatenate((self.state_agent.flatten(), self.state_target.flatten()))

    def _get_info(self):
        pass

    def reset(self, state):
        # We need the following line to seed self.np_random

        # Choose the agent's location uniformly at random

        self.state_agent = np.array([[0, 0], [4, 0]])
        self.state_target = []
        for i in range(5):
            self.state_target.append(state[i])
        self.a_holding = []
        self.state_target = np.asarray(self.state_target)

        #self.baskets = np.array([[0,5],[1,5],[2,5],[3,5],[4,5]])
        self.items = self.state_target.tolist()
        observation = self._get_obs()
        self.step_nb = 0
        self.b_status = [0, 0, 0, 0, 0]
        self.i_status = [0, 0, 0, 0, 0]
        # if self.render_mode == "human":
        #     self._render_frame()

        return observation

    def step(self, action: int):
        # Map the action (element of {0,1,2,3}) to the direction we walk in

        for i, action in enumerate(self.action_map[action]):
            reward = -1
            if action == 3:
                self.state_agent[i, 1] = self.state_agent[i,
                                                          1] - 1 if self.state_agent[i, 1] > 0 else 0
            elif action == 0:
                self.state_agent[i, 0] = self.state_agent[i, 0] + \
                    1 if self.state_agent[i, 0] < self.gridsize - \
                    1 else self.gridsize - 1
            elif action == 1:
                self.state_agent[i, 1] = self.state_agent[i, 1] + \
                    1 if self.state_agent[i,
                                          1] < self.gridsize else self.gridsize
            elif action == 2:
                self.state_agent[i, 0] = self.state_agent[i,
                                                          0] - 1 if self.state_agent[i, 0] > 0 else 0
            elif action == 4:
                self.a_holding.append(self.state_agent[i].tolist())
            elif action == 5:
                self.b_status[self.state_agent[i][0]] = 1

        terminated = False

        # for i in range(2):
        #     if np.all(np.array_equal(self.state_target[i], self.state_agent[i]) and self.status[i]==0):
        #         reward += 10.0
        #         self.status[i] = 1
        # if np.all(self.state_target[0] == self.state_agent[1]):
        #     reward -= 20.0
        #     terminated = True
        # if self.status == [1,1]:
        #     reward += 20.0
        #     terminated = True
        # if self.step_nb == self.nb_steps:
        #     terminated = True
        # info = {}

        # for i in range(2):
        #     if (self.state_agent[i].tolist() in self.items) and self.a_holding[i] == 0:
        #         # print("Picking")
        #         reward += 200.0
        #         self.a_holding[i] = self.state_target.tolist().index(self.state_agent[i].tolist())+1
        #         self.items.remove(self.state_agent[i].tolist())
        #         # print(self.a_holding)

        #     if np.array_equal(self.state_agent[i], [self.a_holding[i]-1,5]) and self.a_holding[i]!= 0:
        #         # print("Dropping")
        #         reward += 200.0
        #         self.b_status[self.a_holding[i]-1] = 1
        #         self.a_holding[i] = 0
        # print(self.b_status)

        if np.all(self.b_status):
            reward += 1000.0
            terminated = True

        if np.all(self.state_agent[0] == self.state_agent[1]):
            reward -= 1000.0
            terminated = True

        self.step_nb += 1
        if self.step_nb == self.nb_steps:
            terminated = True

        observation = self._get_obs()
        if self.debug:
            print("Reward:", reward)
            print('Observation:', observation)

        info = {}

        return observation, reward, terminated, info

    def render(self, mode='human'):
        return self._render_frame()

    BASKET = pygame.image.load('resources/basket.png')
    BASKET = pygame.transform.scale(BASKET, (100, 100))
    ROBOT = pygame.image.load('resources/robot.png')
    ROBOT = pygame.transform.scale(ROBOT, (100, 100))
    tick = pygame.image.load('resources/tick.png')
    tick = pygame.transform.scale(tick, (100, 100))
    cross = pygame.image.load('resources/cross.png')
    cross = pygame.transform.scale(cross, (100, 100))

    def _render_frame(self):
        RED = (255, 143, 143)
        BLUE = (149, 225, 249)
        GREEN = (169, 249, 149)
        PURPLE = (187, 149, 249)
        YELLOW = (249, 209, 149)
        colours = [RED, BLUE, GREEN, PURPLE, YELLOW]

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size+200))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size+200))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )

        for i, target in enumerate(self.state_target):
            if target.tolist() not in self.a_holding:
                rect = pygame.Rect(target[0] * pix_square_size, target[1] * pix_square_size, pix_square_size,
                                   pix_square_size)
                pygame.draw.rect(canvas, colours[i], rect)

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # for agent in self.state_agent:
        #     # Compute center
        #     center = (agent[0] * pix_square_size + pix_square_size / 2,
        #               agent[1] * pix_square_size + pix_square_size / 2)
        #     pygame.draw.circle(canvas, (0,0,255), center, pix_square_size / 3)

        for i in range(5):
            if self.b_status[i] == 0:
                canvas.blit(self.cross, (pix_square_size * np.array([i, 6])))
                rect = pygame.Rect(i * pix_square_size, 5 * pix_square_size, pix_square_size,
                                   pix_square_size)
                pygame.draw.rect(canvas, colours[i], rect)
                canvas.blit(self.cross, (pix_square_size * np.array([i, 6])))
            else:
                canvas.blit(self.tick, (pix_square_size * np.array([i, 6])))
        for i in range(5):
            canvas.blit(self.BASKET, (pix_square_size * np.array([i, 5])))

        for i in range(2):
            canvas.blit(self.ROBOT, ((self.state_agent[i]) * pix_square_size))
        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
