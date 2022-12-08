import sys
import numpy as np
import math
import random
import gym
import warehouse_grid
from itertools import permutations, combinations
import time
from visual_world import *


def shuffle_simulate(agent1_position, agent2_position, agent1_actions, agent2_actions, all_actions1, all_actions2, ix1, ix2, counting):

    find_thredhold1 = 0
    find_thredhold2 = 0
    if agent1_actions.count(4) > 0:
        find_thredhold1 = agent1_actions.index(4)
    elif agent1_actions.count(5) > 0:
        find_thredhold1 = agent1_actions.index(5)

    if agent2_actions.count(4) > 0:
        find_thredhold2 = agent2_actions.index(4)
    elif agent2_actions.count(5) > 0:
        find_thredhold2 = agent2_actions.index(5)
    ag1_action1 = agent1_actions[:find_thredhold1]
    ag2_action2 = agent2_actions[:find_thredhold2]

    shuffle_failing = 0

    # shuffling
    combination1 = list(set(permutations(ag1_action1)))
    combination2 = list(set(permutations(ag2_action2)))
    v_flag = False

    for combination_value in combination1:
        for combination_value2 in combination2:

            temp_combination_value = list(combination_value)
            temp_combination_value2 = list(combination_value2)
            temp_all_actions1 = all_actions1
            temp_all_actions2 = all_actions2

            temp_combination_value.append(agent1_actions[-1])
            temp_combination_value2.append(agent2_actions[-1])
            temp_all_actions1[ix1[0]][ix1[1]] = temp_combination_value
            temp_all_actions2[ix2[0]][ix2[1]] = temp_combination_value2

            ppp1, ppp2, _, _ = validation(agent1_position, agent2_position,
                                          temp_all_actions1, temp_all_actions2, False)

            vv = create_valid_arr(ppp1, ppp2)

            counter2 = vv.count(False)

            if counter2 < counting:
                v_flag = True
                break
            else:
                v_flag = False

        if v_flag:
            # print("========== Replanning success ============")
            combination_value = list(combination_value)
            combination_value2 = list(combination_value2)
            combination_value.append(agent1_actions[-1])
            combination_value2.append(agent2_actions[-1])
            return combination_value, combination_value2, 1

    return 0, 0, shuffle_failing


def fianl_simulation(agent1_position, agent2_position, agent1_actions, agent2_actions, b_items):
    leng1 = len(agent1_actions)
    leng2 = len(agent2_actions)
    mode = "human"
    fps = 4
    if leng1 > leng2:
        iteration = leng1
    else:
        iteration = leng2
    tt1 = np.array([20, 20], dtype=int)
    tt2 = np.array([20, 20], dtype=int)
    state = env.reset(ag1=agent1_position,
                      ag2=agent2_position, t1=tt1, t2=tt2, flag=0, items=b_items)
    for i in range(iteration):
        env.final_render()
        reward = 0
        done = False

        ## agent1 moving ##
        if i >= len(agent1_actions):
            action1 = agent1_actions[-1]
        else:
            action1 = agent1_actions[i]
        if i >= len(agent2_actions):
            action2 = agent2_actions[-1]
        else:
            action2 = agent2_actions[i]

        action = [action1, action2]
        # Do action and get result
        env.step(action)

    return


def simulate(agent1_position, agent2_position, target1, target2, b_items, q_table, ob):

    global epsilon, epsilon_decay
    q_table = q_table
    action_space = []
    best_actions = []
    for r in range(5):
        for c in range(5):
            action_space.append([r, c])

    a1_action_seq = None
    a2_action_seq = None
    ob = ob

    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset(ag1=agent1_position,
                          ag2=agent2_position, t1=target1, t2=target2, flag=1, items=b_items)
        total_reward = 0

        a1_action_seq = []
        a2_action_seq = []
        action_s = []
        for i in range(MAX_TRY):
            # print(i)
            env.render()
            # In the beginning, do random action to learn
            agent1 = state['agent1']
            agent2 = state['agent2']
            new_state = None
            reward = 0
            done = False

            ## agent1 moving ##
            action1 = 0
            action2 = 0
            action = None
            index1_ob = index_2d(ob, agent1)
            index2_ob = index_2d(ob, agent2)

            if random.uniform(0, 1) < epsilon:
                action_ran = random.randint(0, 24)
                action = action_space[action_ran]
            else:
                # print("QQQQQQQ")
                best_action_Q = np.argmax(q_table[index1_ob, index2_ob])
                action = action_space[best_action_Q]

            action1 = action[0]
            action2 = action[1]
            action_s.append(action)
            index_action = index_2d(action_space, action)
            a1_action_seq.append(action1)
            a2_action_seq.append(action2)
            next_state, reward, done, _, = env.step(action)
            total_reward += reward

            # Get correspond q value from state, action pair
            next_agent1 = next_state['agent1']
            next_agent2 = next_state['agent2']

            next_index1_ob = index_2d(ob, next_agent1)
            next_index2_ob = index_2d(ob, next_agent2)
            agent_q_value = q_table[index1_ob, index2_ob][index_action]
            # agent2_q_value = q_table[agent2[0], agent2[1]][action2]
            # q_value = q_table[x_agent, y_agent][action]

            best_q = np.max(q_table[next_index1_ob, next_index2_ob])

            # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
            q_table[index1_ob, index2_ob][index_action] = (
                1 - learning_rate) * agent_q_value + learning_rate * (reward + gamma * best_q)

            state = next_state

            # Draw games
            # env.render()

            # When episode is done, print reward
            if done or i >= MAX_TRY - 1:
                print("Episode %d finished after %i time steps with total reward = %f." % (
                    episode, i, total_reward))
                break
        # exploring rate decay
        if epsilon >= 0.005:
            epsilon -= epsilon_decay
    return q_table


def index_2d(myList, v):
    for i, x in enumerate(myList):
        if np.array_equal(v, x):
            return i


def find_action(actions, index):
    count = 0
    point_index = []
    for r in range(len(actions)):
        for c in range(len(actions[r])):
            for d in range(len(actions[r][c])):
                if count == index:
                    point_index.append(r)
                    point_index.append(c)
                    point_index.append(d)
                    return point_index
                else:
                    count += 1
    return point_index


def action_to_pos(oldposition, actioinflag):
    dir = None
    if actioinflag == 0:  # right
        dir = np.array([1, 0], dtype=int)
    elif actioinflag == 1:  # down
        dir = np.array([0, 1], dtype=int)
    elif actioinflag == 2:  # left
        dir = np.array([-1, 0], dtype=int)
    elif actioinflag == 3:  # up
        dir = np.array([0, -1], dtype=int)
    elif actioinflag == 4:  # picking
        dir = np.array([0, 0], dtype=int)
    elif actioinflag == 5:  # dropping
        dir = np.array([0, 0], dtype=int)
    elif actioinflag == 6:  # waiting
        dir = np.array([0, 0], dtype=int)

    new_pos = oldposition + dir
    new_pos_x = np.clip(new_pos[0], 0, 4)
    new_pos_y = np.clip(new_pos[1], 0, 5)
    return np.array([new_pos_x, new_pos_y])
    # return new_pos


def validation(agent1_pos, agent2_pos, action1, action2, flatten):
    action1_flat = []
    action2_flat = []
    a1_location = np.array(agent1_pos)
    a2_location = np.array(agent2_pos)
    if flatten == False:
        for i in action1:
            for ii in i:
                for iii in ii:
                    action1_flat.append(iii)

        for z in action2:
            for zz in z:
                for zzz in zz:
                    action2_flat.append(zzz)
    else:
        action1_flat = action1
        action2_flat = action2
    pos1_flat = []
    pos2_flat = []

    for w in action1_flat:
        nn = action_to_pos(a1_location, w).tolist()
        pos1_flat.append(nn)
        a1_location = np.array(nn, dtype=int)
    for e in action2_flat:
        ee = action_to_pos(a2_location, e).tolist()
        pos2_flat.append(ee)
        a2_location = ee

    # If the agents finish at the same baskets
    # if len(action1_flat) > len(action2_flat):
    #     if pos2_flat[-1] == pos1_flat[-1]:
    #         if pos2_flat[-1][0] == 4:
    #             agent2_action_sequence[-1][-1].append(2)
    #         elif pos2_flat[-1][0] == 0:
    #             agent2_action_sequence[-1][-1].append(0)
    # else:
    #     if pos2_flat[-1] == pos1_flat[-1]:
    #         if pos1_flat[-1][0] == 4:
    #             agent1_action_sequence[-1][-1].append(2)
    #         elif pos1_flat[-1][0] == 0:
    #             agent1_action_sequence[-1][-1].append(2)

    return pos1_flat, pos2_flat, action1_flat, action2_flat


def create_valid_arr(posarray1, posarray2):
    iteration = 0
    if len(posarray1) < len(posarray2):
        iteration = len(posarray1)
    else:
        iteration = len(posarray2)

    valid_arr = []
    for iter in range(iteration-1):
        if posarray1[iter] == posarray2[iter+1] and posarray1[iter+1] == posarray2[iter]:
            valid_arr.append(False)
        elif posarray1[iter] == posarray2[iter]:
            valid_arr.append(False)
        else:
            valid_arr.append(True)

    return valid_arr


def combination_method(agent1_action_sequence, agent2_action_sequence, a1, a2, p1, p2, v):
    ag1_actions = agent1_action_sequence
    ag2_actions = agent2_action_sequence
    a1 = a1
    a2 = a2
    p1 = p1
    p2 = p2
    v = v
    flag = True
    while False in v:
        count_false = v.count(False)
        collision_index = v.index(False)
        ix1 = find_action(ag1_actions, collision_index)
        ix2 = find_action(ag2_actions, collision_index)
        unvalid_action1 = ag1_actions[ix1[0]][ix1[1]]
        unvalid_action2 = ag2_actions[ix2[0]][ix2[1]]
        # action1_index = ix1[0]*2 + ix1[1]
        # action2_index = ix2[0]*2 + ix2[1]
        flat1_index = 0
        flat2_index = 0
        for i in np.arange(collision_index-1, -1, -1):
            if a1[i] == 4 or a1[i] == 5:
                flat1_index = i
                break
        for i in np.arange(collision_index-1, -1, -1):
            if a2[i] == 4 or a2[i] == 5:
                flat2_index = i
                break

        # print('========== go to replanning =====')
        v_action1, v_action2, conv = shuffle_simulate(agent1_pos, agent2_pos, unvalid_action1,
                                                      unvalid_action2, ag1_actions, ag2_actions, ix1, ix2, count_false)

        if conv == 1:
            # print('========== find optimal combination ==========')
            ag1_actions[ix1[0]][ix1[1]] = v_action1
            ag2_actions[ix2[0]][ix2[1]] = v_action2
            flag = True
        else:
            # print("==========it is failing case==========")
            flag = False
            break

        p1, p2, a1, a2 = validation(agent1_pos, agent2_pos,
                                    ag1_actions, ag2_actions, False)

        v = create_valid_arr(p1, p2)
        # check_inf = v.count(False)
        # print(len(p1))
        # print(len(p2))
        # print(len(v))
        if v.count(False) == 0:
            flag = True
            break

    if flag == False:
        return 0, 0, 0
    else:
        return ag1_actions, ag2_actions, 1


def q_learning_method(agent1_action_sequence, agent2_action_sequence, a1, a2, p1, p2, v):
    ag1_actions = agent1_action_sequence
    ag2_actions = agent2_action_sequence
    a1 = a1
    a2 = a2
    p1 = p1
    p2 = p2
    v = v
    flag = True
    ob = []
    q_table = []
    # print(agent1_action_sequence)
    # print(agent2_action_sequence)
    # print(p1)
    # print(p2)
    # print(v)
    print("q_learning_method")
    while False in v:
        print("false counter", v.count(False))
        # Q-learning table
        best_actions = []
        flag = False
        ob = []
        for a in range(5):
            for b in range(6):
                ob.append([a, b])

        q_table = np.zeros((30, 30, 25))

        for r in range(5):
            for c in range(5):
                action_space.append([r, c])

        collision_index = v.index(False)
        ix1 = find_action(ag1_actions, collision_index)
        ix2 = find_action(ag2_actions, collision_index)
        unvalid_action1 = ag1_actions[ix1[0]][ix1[1]]
        unvalid_action2 = ag2_actions[ix2[0]][ix2[1]]
        flat1_index = 0
        flat2_index = 0
        for i in np.arange(collision_index-1, -1, -1):
            if a1[i] == 4 or a1[i] == 5:
                flat1_index = i
                break
        for i in np.arange(collision_index-1, -1, -1):
            if a2[i] == 4 or a2[i] == 5:
                flat2_index = i
                break
        temp_a1_pos = np.array(p1[flat1_index], dtype=int)
        temp_a2_pos = np.array(p2[flat2_index], dtype=int)
        t_a1 = a1[flat1_index+1:]
        t_a2 = a2[flat2_index+1:]
        t_p1 = p1[flat1_index+1:]
        t_p2 = p2[flat2_index+1:]
        target1_index = 0
        target2_index = 0
        for i in range(len(t_a1)):
            if t_a1[i] == 4 or t_a1[i] == 5:
                target1_index = i
                break
        for i in range(len(t_a2)):
            if t_a2[i] == 4 or t_a2[i] == 5:
                target2_index = i
                break
        target1 = np.array(t_p1[target1_index], dtype=int)
        target2 = np.array(t_p2[target2_index], dtype=int)
        q_table = simulate(agent1_position=temp_a1_pos, agent2_position=temp_a2_pos,
                           target1=target1, target2=target2, b_items=b_items, q_table=q_table, ob=ob)

        actions = []
        checker1 = np.array_equal(temp_a1_pos, target1)
        checker2 = np.array_equal(temp_a2_pos, target2)

        count_try = 0
        while checker1 is False or checker2 is False:
            index1_ob = index_2d(ob, temp_a1_pos)
            index2_ob = index_2d(ob, temp_a2_pos)
            best_action = np.argmax(q_table[index1_ob, index2_ob])
            action = action_space[best_action]
            actions.append(action)
            action_1 = action[0]
            action_2 = action[1]
            temp_a1_pos = action_to_pos(temp_a1_pos, action_1)
            temp_a2_pos = action_to_pos(temp_a2_pos, action_2)
            checker1 = np.array_equal(temp_a1_pos, target1)
            checker2 = np.array_equal(temp_a2_pos, target2)
            count_try += 1
            # print(temp_a1_pos)
            # print(temp_a2_pos)
            # time.sleep(0.2)
            if count_try == 100:
                print("--false---")
                print("The Q-table doesnt converge")
                print(target1)
                print(target2)
                return 0, 0, 0
                break

        aa1 = []
        aa2 = []
        for i in actions:
            if i[0] == 4:
                aa1.append(6)
            else:
                aa1.append(i[0])
            if i[1] == 4:
                aa2.append(6)
            else:
                aa2.append(i[1])

        if np.array_equal(target1, target2):
            if len(aa1) < len(aa2):
                aa1.insert(0, 6)
                aa1.insert(0, 6)
            else:
                aa2.insert(0, 6)
                aa2.insert(0, 6)

        aa1.append(unvalid_action1[-1])
        aa2.append(unvalid_action2[-1])

        ag1_actions[ix1[0]][ix1[1]] = aa1
        ag2_actions[ix2[0]][ix2[1]] = aa2

        p1, p2, a1, a2 = validation(agent1_pos, agent2_pos,
                                    ag1_actions, ag2_actions, False)

        v = create_valid_arr(p1, p2)
        checker = v.count(False)
        if checker == 0:
            flag = True
            break

    if flag == False:
        return 0, 0, 0
    else:
        return ag1_actions, ag2_actions, 1


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

    print("Enter the replanning mode ")
    print("0 = combination")
    print("1 = q-learning")
    replanning_mode = input("Enter : ")
    replanning_mode = int(replanning_mode)
    env = gym.make("warehouse_grid/GridWorld-v0")
    b_items = env.get_basket_items()

    ###### Q -learning####
    MAX_EPISODES = 3000
    MAX_TRY = 100
    epsilon = 0.99
    epsilon_decay = 0.0003
    learning_rate = 0.1
    gamma = 0.9

    agent1_action_sequence, agent2_action_sequence = env.get_action_sequence()
    agent1_pos, agent2_pos = env.get_agents_location()
    init_pos = env.get_agents_initial_location()

    p1, p2, a1, a2 = validation(agent1_pos, agent2_pos,
                                agent1_action_sequence, agent2_action_sequence, False)
    v = create_valid_arr(p1, p2)
    temp_a1_pos = 0
    temp_a2_pos = 0
    collision_action1_seq = []
    collision_action2_seq = []
    target1 = None
    target2 = None

    valid_action1_seq = None
    valid_action2_seq = None
    val_count = v.count(False)
    action_space = []
    q_table = []
    ob = []
    items = []

    for i in b_items:
        items.append(i[0].tolist())

    if val_count != 0:
        if replanning_mode == 0:
            ag1_valid_actions, ag2_valid_actions, v_flag = combination_method(
                agent1_action_sequence, agent2_action_sequence, a1, a2, p1, p2, v)

        elif replanning_mode == 1:
            ag1_valid_actions, ag2_valid_actions, v_flag = q_learning_method(
                agent1_action_sequence, agent2_action_sequence, a1, a2, p1, p2, v)
        else:
            pass
        if v_flag == 0:
            print("========== there is no solution ==========")
            print(a1)
            print(a2)
            print(p1)
            print(p2)

        else:
            print("========== replanning success ==========")
            p1, p2, a1, a2 = validation(agent1_pos, agent2_pos,
                                        ag1_valid_actions, ag2_valid_actions, False)
            print("Agent 1 action sequence")
            print(a1)
            print("Agent 2 action sequence")
            print(a2)
            print(b_items)
            v = create_valid_arr(p1, p2)

    else:
        print("====== Action planning finds optimal paths")
        print("Agent 1 action sequence")
        print(a1)
        print("Agent 2 action sequence ")
        print(a2)
        print(b_items)

    visual_sequence = make_sequence(a1, a2)
    env = VisWorldEnv()
    env.reset(items)
    for action in visual_sequence:
        env.step(action)
        env.render()

    # fianl_simulation(agent1_pos, agent2_pos, a1, a2, b_items=b_items)
