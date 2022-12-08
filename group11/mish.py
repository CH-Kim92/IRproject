import sys
import numpy as np
import math
import random
import gym
from visual_world import *


def make_sequence(a1, a2):
    action_map = {}
    for i in range(49):
        # Change i to base 7 (possible actions)
        action = [0] * 2  # for 2 agents
        num = i
        index = -1
        while num > 0:
            action[index] = num % 7  # number of actions 5
            num = num // 7  # number of actions 5
            index -= 1
        action_map[i] = action

    key_list = list(action_map.keys())
    val_list = list(action_map.values())

    if len(a1) < len(a2):
        for i in range(len(a2)-len(a1)):
            a1.append(6)
    else:
        for i in range(len(a1)-len(a2)):
            a2.append(6)

    action_pair = []
    for i in range(len(a1)):
        action_pair.append([a1[i], a2[i]])

    action_sequence = [key_list[val_list.index(i)] for i in action_pair]
    return action_sequence


if __name__ == "__main__":
    actS = make_sequence([1, 1, 1, 4, 1, 1, 0, 0, 0, 5, 3, 4, 0, 1, 6, 1, 1, 1, 5, 3, 3, 3, 4, 1, 1, 1, 2, 2, 2, 5], [
                         1, 4, 2, 2, 2, 2, 1, 1, 1, 1, 5, 3, 0, 0, 3, 0, 0, 4, 2, 2, 1, 1, 5])
    env = VisWorldEnv()
    env.reset([[4, 1], [4, 2], [4, 3], [0, 3], [3, 4]])
    print(actS)
    for i in actS:
        env.step(i)
        env.render()
