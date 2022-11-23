import sys
import numpy as np
import math
import random
import gym
import warehouse_grid
from itertools import permutations, combinations


def shuffle_simulate(agent1_position, agent2_position, agent1_actions, agent2_actions, target1, target2):
    global epsilon, epsilon_decay
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
    ac1_len = len(ag1_action1)
    ac2_len = len(ag2_action2)
    iter = 0
    cc = 0
    if ac1_len < ac2_len:
        iter = ac2_len
        for _ in range(ac2_len-ac1_len):
            ag1_action1.append(6)
    elif ac1_len > ac2_len:
        iter = ac1_len
        for _ in range(ac1_len-ac2_len):
            ag2_action2.append(6)
    else:
        iter = ac1_len
    a1 = list(set(permutations(ag1_action1)))
    a2 = list(set(permutations(ag2_action2)))

    for p1 in a1:
        for p2 in a2:
            # print(p1, p2)
            state = env.reset(ag1=agent1_position,
                              ag2=agent2_position, t1=target1, t2=target2, flag=1)
            for i in range(iter):
                action1 = p1[i]
                action2 = p2[i]
                action = [action1, action2]
                obs, reward, done, _, = env.step(action)
                env.render()
                if np.array_equal(obs['agent1'], target1) and np.array_equal(obs['agent2'], target2):
                    # print("hi")
                    aa = list(p1)
                    bb = list(p2)
                    aa.append(agent1_actions[-1])
                    bb.append(agent2_actions[-1])
                    cc = 1
                    return aa, bb, cc
                if done:
                    break
    return 0, 0, cc


def fianl_simulation(agent1_position, agent2_position, agent1_actions, agent2_actions):
    leng1 = len(agent1_actions)
    leng2 = len(agent2_actions)
    if leng1 > leng2:
        iteration = leng1
    else:
        iteration = leng2
    # aaa = np.array(agent1_position,dtype=int)
    # bbb = np.array(agent2_position,dtype=int)
    tt1 = np.array([20, 20], dtype=int)
    tt2 = np.array([20, 20], dtype=int)
    state = env.reset(ag1=agent1_position,
                      ag2=agent2_position, t1=tt1, t2=tt2, flag=0)
    for i in range(iteration):

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

        # Draw games
        env.render()

        # print('episode :', episode)
        # When episode is done, print reward
        # if done or i >= MAX_TRY - 1:
        #     break

    return


def simulate(agent1_position, agent2_position, agent1_actions, agent2_actions, target1, target2):
    global epsilon, epsilon_decay
    action_space = []
    best_actions = []
    for r in range(5):
        for c in range(5):
            action_space.append([r, c])
    a1_action_seq = None
    a2_action_seq = None

    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset(ag1=agent1_position,
                          ag2=agent2_position, t1=target1, t2=target2, flag=1)
        total_reward = 0

        a1_action_seq = []
        a2_action_seq = []
        action_s = []
        for i in range(MAX_TRY):

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
            # print(agent1)
            index1_ob = index_2d(ob, agent1)
            index2_ob = index_2d(ob, agent2)

            if random.uniform(0, 1) < epsilon:
                action_ran = random.randint(0, 24)
                if np.array_equal(agent1, target1):
                    action_ran = random.randint(19, 24)
                    action = action_space[action_ran]
                elif np.array_equal(agent2, target2):
                    inx = [x-1 for x in range(25) if x % 5 == 0]
                    inx.remove(-1)
                    aa = random.choice(inx)
                    action = action_space[aa]
                else:
                    action = action_space[action_ran]
            else:
                action1 = np.argmax(q_table[index1_ob, index2_ob])
                action = action_space[action1]

            action_s.append(action)
            index_action = index_2d(action_space, action)
            a1_action_seq.append(action1)
            a2_action_seq.append(action2)
            # Do action and get result
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
            env.render()

            # When episode is done, print reward
            if done or i >= MAX_TRY - 1:
                print("Episode %d finished after %i time steps with total reward = %f." % (
                    episode, i, total_reward))
                break
        # exploring rate decay
        if epsilon >= 0.005:
            epsilon -= epsilon_decay
    return i


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

    new_position = oldposition + dir
    return new_position


def validation(agent1_pos, agent2_pos, action1, action2):
    action1_flat = []
    action2_flat = []
    a1_location = np.array(agent1_pos)
    a2_location = np.array(agent2_pos)
    for i in action1:
        for ii in i:
            for iii in ii:
                action1_flat.append(iii)

    for z in action2:
        for zz in z:
            for zzz in zz:
                action2_flat.append(zzz)

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
    if len(action1_flat) > len(action2_flat):
        if pos2_flat[-1] == pos1_flat[-1]:
            if pos2_flat[-1][0] == 4:
                agent2_action_sequence[-1][-1].append(2)
            elif pos2_flat[-1][0] == 0:
                agent2_action_sequence[-1][-1].append(0)
    else:
        if pos2_flat[-1] == pos1_flat[-1]:
            if pos1_flat[-1][0] == 4:
                agent1_action_sequence[-1][-1].append(2)
            elif pos1_flat[-1][0] == 0:
                agent1_action_sequence[-1][-1].append(2)

    return pos1_flat, pos2_flat, action1_flat, action2_flat


def create_valid_arr(posarray1, posarray2):
    iteration = 0
    if len(posarray1) < len(posarray2):
        iteration = len(posarray1)
    else:
        iteration = len(posarray2)

    valid_arr = []
    for iter in range(iteration):
        if posarray1[iter] == posarray2[iter]:
            valid_arr.append(False)
        else:
            valid_arr.append(True)

    return valid_arr


if __name__ == "__main__":
    env = gym.make("warehouse_grid/GridWorld-v0")
    SUFFLE_MAX_EPISODES = 1000

    ###### Q -learning####
    MAX_EPISODES = 20
    MAX_TRY = 100
    epsilon = 0.99
    epsilon_decay = 0.00001
    learning_rate = 0.2
    gamma = 0.9
    ####### Q - learning ################

    agent1_action_sequence, agent2_action_sequence = env.get_action_sequence()
    agent1_pos, agent2_pos = env.get_agents_location()
    init_pos = env.get_agents_initial_location()

    p1, p2, a1, a2 = validation(agent1_pos, agent2_pos,
                                agent1_action_sequence, agent2_action_sequence)
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
    ########################### Using Q - Table ########################################
    while False in v:
        # Q-learning table
        best_actions = []
        ob = []
        for a in range(5):
            for b in range(6):
                ob.append([a, b])

        q_table = np.zeros((30, 30, 25))

        for r in range(5):
            for c in range(5):
                action_space.append([r, c])

        collision_index = v.index(False)
        ix1 = find_action(agent1_action_sequence, collision_index)
        ix2 = find_action(agent2_action_sequence, collision_index)
        unvalid_action1 = agent1_action_sequence[ix1[0]][ix1[1]]
        unvalid_action2 = agent2_action_sequence[ix2[0]][ix2[1]]
        action1_index = ix1[0]*2 + ix1[1]
        action2_index = ix2[0]*2 + ix2[1]
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
        if p1[flat1_index + len(unvalid_action1)] == 4 or p1[flat1_index + len(unvalid_action1)] == 5:
            target1 = np.array(
                p1[flat1_index + len(unvalid_action1)], dtype=int)
        else:
            target1 = np.array(
                p1[flat1_index + len(unvalid_action1)-1], dtype=int)
        if p2[flat2_index + len(unvalid_action2)] == 4 or p2[flat2_index + len(unvalid_action2)] == 5:
            target2 = np.array(
                p2[flat2_index + len(unvalid_action2)], dtype=int)
        else:
            target2 = np.array(
                p2[flat2_index + len(unvalid_action2)-1], dtype=int)

        maxiter = simulate(temp_a1_pos, temp_a2_pos, unvalid_action1,
                           unvalid_action2, target1, target2)
        print(q_table)
        # for poz in range(maxiter):
        #     actionidx = np.argmax(q_table[temp_a1_pos][temp_a2_pos])
        #     best_actions.append(action_space[actionidx])
        # if conv == 1:
        #     agent1_action_sequence[ix1[0]][ix1[1]] = v_action1
        #     agent2_action_sequence[ix2[0]][ix2[1]] = v_action2

        p1, p2, a1, a2 = validation(agent1_pos, agent2_pos,
                                    agent1_action_sequence, agent2_action_sequence)
        v = create_valid_arr(p1, p2)
        break

    ################################# Combination of list ###################################
    # print(valid_seq)
    while False in v:
        break
        collision_index = v.index(False)
        ix1 = find_action(agent1_action_sequence, collision_index)
        ix2 = find_action(agent2_action_sequence, collision_index)
        unvalid_action1 = agent1_action_sequence[ix1[0]][ix1[1]]
        unvalid_action2 = agent2_action_sequence[ix2[0]][ix2[1]]
        action1_index = ix1[0]*2 + ix1[1]
        action2_index = ix2[0]*2 + ix2[1]
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
        if p1[flat1_index + len(unvalid_action1)] == 4 or p1[flat1_index + len(unvalid_action1)] == 5:
            target1 = np.array(
                p1[flat1_index + len(unvalid_action1)], dtype=int)
        else:
            target1 = np.array(
                p1[flat1_index + len(unvalid_action1)-1], dtype=int)
        if p2[flat2_index + len(unvalid_action2)] == 4 or p2[flat2_index + len(unvalid_action2)] == 5:
            target2 = np.array(
                p2[flat2_index + len(unvalid_action2)], dtype=int)
        else:
            target2 = np.array(
                p2[flat2_index + len(unvalid_action2)-1], dtype=int)

        v_action1, v_action2, conv = shuffle_simulate(temp_a1_pos, temp_a2_pos, unvalid_action1,
                                                      unvalid_action2, target1, target2)

        if conv == 1:
            agent1_action_sequence[ix1[0]][ix1[1]] = v_action1
            agent2_action_sequence[ix2[0]][ix2[1]] = v_action2

        p1, p2, a1, a2 = validation(agent1_pos, agent2_pos,
                                    agent1_action_sequence, agent2_action_sequence)
        v = create_valid_arr(p1, p2)

    # p1, p2, a1, a2 = validation(agent1_pos, agent2_pos,
    #                             agent1_action_sequence, agent2_action_sequence)
    # fianl_simulation(agent1_pos, agent2_pos, a1, a2)
    # print(agent1_action_sequence)
    # print(agent2_action_sequence)
    # print(a1)
    # print(a2)
    # print(p1)
    # print(p2)
    # print(valid_seq)
    # # print(q_table)
