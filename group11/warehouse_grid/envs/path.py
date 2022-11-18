import numpy as np

"""
input : current agents location , goal lists 
output : action space sequence 
"""


def cur_to_item(current_location, goal):
    action = []
    sub = goal[1]-current_location
    right_left = sub[0]
    up_down = sub[1]

    if up_down < 0:
        for _ in range(abs(up_down)):
            action.append(3)
    if right_left < 0:
        for _ in range(abs(right_left)):
            action.append(2)
    if right_left > 0:
        for _ in range(right_left):
            action.append(0)
    if up_down > 0:
        for _ in range(up_down):
            action.append(1)

    action.append(4)
    return action


def item_to_basket(goal):
    action = []
    sub = goal[0] - goal[1]
    right_left = sub[0]
    up_down = sub[1]

    if up_down < 0:
        for _ in range(abs(up_down)):
            action.append(3)
    if right_left < 0:
        for _ in range(abs(right_left)):
            action.append(2)
    if right_left > 0:
        for _ in range(right_left):
            action.append(0)
    if up_down > 0:
        for _ in range(up_down):
            action.append(1)
    action.append(5)
    return action


def path_planning(current_location, goal_list):
    sp_action = []
    k = 0
    cur_location = current_location
    for i in goal_list:
        first_goal = i
        to_item = cur_to_item(cur_location, first_goal)
        to_basket = item_to_basket(first_goal)
        action = [to_item, to_basket]
        sp_action.append(action)
        k += 1
        if k > len(goal_list)-1:
            break
        cur_location = goal_list[k-1][0]
    return sp_action
