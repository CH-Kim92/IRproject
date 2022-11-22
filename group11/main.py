import sys
import numpy as np
import math
import random
import gym
import warehouse_grid


def shuffle_simulate(agent1_position, agent2_position, agent1_actions, agent2_actions, target1, target2):
    global epsilon, epsilon_decay
    ag1_action1 = agent1_actions[:-1]
    ag2_action2 = agent2_actions[:-1]
    ac1_len = len(ag1_action1)
    ac2_len = len(ag2_action2)
    iteration = 0

    if ac1_len < ac2_len:
        iteration = ac1_len
    else:
        iteration = ac2_len

    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset(ag1=agent1_position,
                          ag2=agent2_position, t1=target1, t2=target2)
        total_reward = 0
        best_action_sequence = []

        ac1 = ag1_action1
        ac2 = ag2_action2

        random.shuffle(ac1)
        random.shuffle(ac2)
        # print(state)

        for i in range(iteration):

            reward = 0
            done = False

            ## agent1 moving ##
            action1 = ac1[i]
            action2 = ac2[i]

            action = [action1, action2]
            # Do action and get result
            _, reward, done, _, = env.step(action)
            total_reward += reward

            # Draw games
            env.render()

            if i == iteration-1:
                ac1.append(agent1_actions[-1])
                ac2.append(agent2_actions[-1])
                return ac1, ac2

            # When episode is done, print reward
            if done or i >= MAX_TRY - 1:
                break

        # exploring rate decay
        if epsilon >= 0.005:
            epsilon *= epsilon_decay

    return


def simulate(agent1_position, agent2_position, agent1_actions, agent2_actions, target1, target2):
    global epsilon, epsilon_decay
    ag1_action1 = agent1_actions[:-1]
    ag2_action2 = agent2_actions[:-1]
    ac1_len = len(ag1_action1)
    ac2_len = len(ag2_action2)
    # print(ag1_action1)
    # print(ag2_action2)
    iteration = 0

    if ac1_len < ac2_len:
        iteration = ac1_len
    else:
        iteration = ac2_len

    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset(ag1=agent1_position,
                          ag2=agent2_position, t1=target1, t2=target2)
        total_reward = 0
        best_action_sequence = []

        ac1 = ag1_action1
        ac2 = ag2_action2

        random.shuffle(ac1)
        random.shuffle(ac2)
        # print(state)
        # AI tries up to MAX_TRY times
        for i in range(iteration):

            # In the beginning, do random action to learn
            agent1 = state['agent1']
            agent2 = state['agent2']
            new_state = None
            reward = 0
            done = False

            ## agent1 moving ##
            action1 = ac1[i]
            action2 = ac2[i]

            action = [action1, action2]
            # Do action and get result
            next_state, reward, done, _, = env.step(action)
            total_reward += reward

            # Get correspond q value from state, action pair
            next_agent1 = next_state['agent1']
            next_agent2 = next_state['agent2']
            agent1_q_value = agent1_q_table[agent1[0], agent1[1]][action1]
            agent2_q_value = agent2_q_table[agent2[0], agent2[1]][action2]
            # q_value = q_table[x_agent, y_agent][action]
            print(action1)
            print(action2)
            print(next_agent1)
            print(next_agent2)
            best_q1 = np.max(agent1_q_table[next_agent1[0], next_agent1[1]])
            best_q2 = np.max(agent2_q_table[next_agent2[0], next_agent2[1]])
            # best_q = np.max(q_table[next_agent_x, next_agent_y])

            # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
            agent1_q_table[agent1[0], agent1[1]][action] = (
                1 - learning_rate) * agent1_q_value + learning_rate * (reward + gamma * best_q1)
            agent2_q_table[agent2[0], agent2[1]][action] = (
                1 - learning_rate) * agent2_q_value + learning_rate * (reward + gamma * best_q2)

            # q_table[x_agent, y_agent][action] = (
            #     1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

            # Set up for the next iteration
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
            epsilon *= epsilon_decay


def check_action(pose, action):
    pose = pose
    if type(pose) is not list:
        pose = pose.tolist()
    if action == 0:  # right
        pose[0] += 1
        return pose
    elif action == 1:  # down
        pose[1] += 1
        return pose
    elif action == 2:  # left
        pose[0] -= 1
        return pose
    elif action == 3:  # up
        pose[1] -= 1
        return pose
    ## pick up ##
    else:
        return pose


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


def posjunk(agent1_pos, agent2_pos, action1, action2):

    # action1 = [j for sub in action1 for j in sub]
    # action2 = [j for sub in action2 for j in sub]
    agent1_pos_sequence = []
    agent2_pos_sequence = []
    pos1 = agent1_pos
    pos2 = agent2_pos
    for i in action1:
        position = check_action(pos1, i)
        # print(type(position))
        agent1_pos_sequence += position
        # print(position)
        pos1 = position
    for ii in action2:
        position2 = check_action(pos2, ii)
        agent2_pos_sequence += position2
        pos2 = position2

    al = int(len(agent1_pos_sequence))
    bl = int(len(agent2_pos_sequence))

    agent1_pos_sequence = np.array(
        agent1_pos_sequence, dtype=int).reshape((int(al/2), 2))
    agent2_pos_sequence = np.array(
        agent2_pos_sequence, dtype=int).reshape((int(bl/2), 2))

    # if al > bl:
    #     for i in range(int(bl/2)):
    #         if np.array_equal(agent1_pos_sequence[i], agent2_pos_sequence[i]):
    #             valid = False
    #             break
    #         else:
    #             valid = True
    # else:
    #     for i in range(int(al/2)):
    #         if np.array_equal(agent1_pos_sequence[i], agent2_pos_sequence[i]):
    #             valid = False
    #             break
    #         else:
    #             valid = True
    # print(agent1_pos_sequence, agent2_pos_sequence)
    return agent1_pos_sequence, agent2_pos_sequence


def valid_concatenate(agent1_pos, agent2_pos, action1, action2):
    pos1 = agent1_pos
    pos2 = agent2_pos
    action1 = action1
    action2 = action2
    al = len(action1)
    bl = len(action2)
    val_seq = []
    small_actions = None

    a1_seq = []
    a2_seq = []
    if al < bl:
        small_actions = al
    else:
        small_actions = bl

    for i in range(small_actions):
        for k in range(2):
            sub_action1 = action1[i][k]
            sub_action2 = action2[i][k]
            # print(sub_action1)
            # print(sub_action2)
            a1_pos, a2_pos = posjunk(
                pos1, pos2, sub_action1, sub_action2)
            a1_seq.append(a1_pos)
            a2_seq.append(a2_pos)
            # val_seq.append(valid)
            pos1 = a1_pos[len(a1_pos)-1]
            pos2 = a2_pos[len(a2_pos)-1]
            # print(valid)

    # print(a1_pos)
    new_a1_seq = []
    new_a2_seq = []
    for i in a1_seq:
        c = i.flatten()
        for s in c:
            new_a1_seq.append(s)
    for i in a2_seq:
        c = i.flatten()
        for s in c:
            new_a2_seq.append(s)
    l1 = len(new_a1_seq)
    l2 = len(new_a2_seq)
    less_action_agent = 0
    smlen = 0
    if l1 < l2:
        less_action_agent = 1
        smlen = l1
    else:
        less_action_agent = 2
        smlen = l2

    for i in range(int(smlen/2)):
        if new_a1_seq[2*i] == new_a2_seq[2*i] and new_a1_seq[2*i+1] == new_a2_seq[2*i+1]:
            valid = False
            val_seq.append(valid)
        else:
            valid = True
            val_seq.append(valid)

    return a1_seq, a2_seq, val_seq, less_action_agent


if __name__ == "__main__":
    env = gym.make("warehouse_grid/GridWorld-v0")
    MAX_EPISODES = 200
    MAX_TRY = 100
    epsilon = 0.9
    epsilon_decay = 0.8
    learning_rate = 0.1
    gamma = 0.6
    agent1_action_sequence, agnet2_action_sequence = env.get_action_sequence()
    agent1_pos, agent2_pos = env.get_agents_location()
    a1_seq, a2_seq, valid_seq, less_action_agent = valid_concatenate(agent1_pos, agent2_pos,
                                                                     agent1_action_sequence, agnet2_action_sequence)
    if np.array_equal(a1_seq[-1][-1], a2_seq[-1][-1]):
        if less_action_agent == 1:
            new_ac = a1_seq[-1][-1]
            if new_ac[0] == 4:
                agent1_action_sequence[-1][-1].append(2)
            else:
                agent1_action_sequence[-1][-1].append(0)
        else:
            new_ac = a2_seq[-1][-1]
            if new_ac[0] == 4:
                agent1_action_sequence[-1][-1].append(2)
            else:
                agent1_action_sequence[-1][-1].append(0)

    agent1_q_table = np.zeros((6, 6, 4))
    agent2_q_table = np.zeros((6, 6, 4))
    # q_table = np.zeros((6, 6, 6))

    temp_a1_pos = 0
    temp_a2_pos = 0
    collision_action1_seq = []
    collision_action2_seq = []
    target1 = None
    target2 = None

    if valid_seq.count(False) != 0:
        collistion_index = valid_seq.index(False)
        ix1 = find_action(agent1_action_sequence, collistion_index)
        ix2 = find_action(agnet2_action_sequence, collistion_index)
        # print(a1_seq)
        # print(a2_seq)
        unvalid_action1 = agent1_action_sequence[ix1[0]][ix1[1]]
        unvalid_action2 = agnet2_action_sequence[ix2[0]][ix2[1]]
        # print(ix2)
        # print(agnet2_action_sequence)
        action1_index = ix1[0]*2 + ix1[1]
        action2_index = ix2[0]*2 + ix2[1]
        temp_a1_pos = a1_seq[action1_index - 1][-1]
        temp_a2_pos = a2_seq[action2_index - 1][-1]
        target1 = a1_seq[action1_index][-1]
        target2 = a2_seq[action2_index][-1]
        v_action1, v_action2 = shuffle_simulate(temp_a1_pos, temp_a2_pos, unvalid_action1,
                                                unvalid_action2, target1, target2)

        agent1_action_sequence[ix1[0]][ix1[1]] = v_action1
        agnet2_action_sequence[ix2[0]][ix2[1]] = v_action2

    a1_seq, a2_seq, valid_seq, less_action_agent = valid_concatenate([0, 0], [3, 0],
                                                                     agent1_action_sequence, agnet2_action_sequence)

    # print(a1_seq[ix1[0]][ix1[1]])

    # if valid_seq.count(False) != 0:
    #     collistion_index = valid_seq.index(False)
    #     # print(collistion_index)
    #     # print(len(agent1_action_sequence))
    #     if collistion_index == 0:
    #         temp_a1_pos = agent1_pos
    #         temp_a2_pos = agent2_pos
    #     else:
    #         temp_a1_pos = a1_seq[collistion_index-1][-1]
    #         temp_a2_pos = a2_seq[collistion_index-1][-1]
    #     if collistion_index % 2 == 0:
    #         id = 0
    #     else:
    #         id = 1
    #     collision_action1_seq = agent1_action_sequence[int(
    #         collistion_index/2)][id]
    #     collision_action2_seq = agnet2_action_sequence[int(
    #         collistion_index/2)][id]
    #     target1 = a1_seq[collistion_index][-1]
    #     target2 = a2_seq[collistion_index][-1]
    # print(temp_a1_pos)
    # print(temp_a2_pos)
    # print(collision_action1_seq)
    # print(collision_action2_seq)
    # print(target1)
    # print(target2)
    # simulate(temp_a1_pos, temp_a2_pos,
    #          collision_action1_seq, collision_action2_seq, target1, target2)

    # collision_index = valid_seq.index(False)
    # temp_agent1_location =
    # simulate()
