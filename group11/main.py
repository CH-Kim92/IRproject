import sys
import numpy as np
import math
import random
import gym
import warehouse_grid


def simulate(agent1_position, agent2_position, agent1_actions, agent2_actions, target1, target2):
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
        if ac1_len < ac2_len:
            iteration = ac1_len
        else:
            iteration = ac2_len

        # Init environment
        state = env.reset(ag1=agent1_position,
                          ag2=agent2_position, t1=target1, t2=target2)
        total_reward = 0
        best_action_sequence = []

        ac1 = ag1_action1
        ac2 = ag2_action2

        random.shuffle(ac1)
        random.shuffle(ac2)

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
            # for i in range(num_action):
            # if random.uniform(0, 1) < epsilon and y_agent == 5:
            #     ls = [0, 1, 2, 3, 5]
            #     action = random.choice(ls)
            # elif random.uniform(0, 1) < epsilon and y_agent < 5:
            #     ls = [0, 1, 2, 3, 4]
            #     action = random.choice(ls)
            # else:
            #     if y_agent == 5:
            #         ls = q_table[x_agent, y_agent]
            #         ls[4] = min(ls)
            #         action = np.argmax(ls)
            #     elif y_agent < 5:
            #         ls = q_table[x_agent, y_agent]
            #         ls[5] = min(ls)
            #         action = np.argmax(ls)

            # Do action and get result
            next_state, reward, done, _, = env.step(action)
            total_reward += reward

            # Get correspond q value from state, action pair
            next_agent1 = next_state['agent1']
            next_agent2 = next_state['agent2']
            agent1_q_value = agent1_q_table[agent1[0], agent1[1]][action]
            agent2_q_value = agent2_q_table[agent2[0], agent2[1]][action]
            # q_value = q_table[x_agent, y_agent][action]
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


def validation(agent1_pos, agent2_pos, action1, action2):
    valid = False
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

    if al > bl:
        for i in range(int(bl/2)):
            if np.array_equal(agent1_pos_sequence[i], agent2_pos_sequence[i]):
                valid = False
                break
            else:
                valid = True
    else:
        for i in range(int(al/2)):
            if np.array_equal(agent1_pos_sequence[i], agent2_pos_sequence[i]):
                valid = False
                break
            else:
                valid = True
    # print(agent1_pos_sequence, agent2_pos_sequence)
    return agent1_pos_sequence, agent2_pos_sequence, valid


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
            a1_pos, a2_pos, valid = validation(
                pos1, pos2, sub_action1, sub_action2)
            a1_seq.append(a1_pos)
            a2_seq.append(a2_pos)
            val_seq.append(valid)
            pos1 = a1_pos[len(a1_pos)-1]
            pos2 = a2_pos[len(a2_pos)-1]
            # print(valid)
    l1 = len(a1_seq)
    l2 = len(a2_seq)
    # a1_seq = np.array(a1_seq)
    # a2_seq = np.array(a2_seq)
    # a1_seq = a1_seq.flatten()
    print(a1_seq)
    smlen = 0
    if l1 < l2:
        smlen = l1
    else:
        smeln = l2
    return a1_seq, a2_seq, val_seq


if __name__ == "__main__":
    env = gym.make("warehouse_grid/GridWorld-v0")
    MAX_EPISODES = 200
    MAX_TRY = 100
    epsilon = 0.9
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6
    agent1_action_sequence, agnet2_action_sequence = env.get_action_sequence()
    # print(agent1_action_sequence)
    # print(agnet2_action_sequence)
    agent1_pos, agent2_pos = env.get_agents_location()
    # print(validation(agent1_pos, agent2_pos,
    #       agent1_action_sequence[0][1], agnet2_action_sequence[0][1]))
    a1_seq, a2_seq, valid_seq = valid_concatenate(agent1_pos, agent2_pos,
                                                  agent1_action_sequence, agnet2_action_sequence)
    # print(a1_seq)  # position
    # print(a2_seq)  # position
    # print(valid_seq)
    agent1_q_table = np.zeros((6, 6, 4))
    agent2_q_table = np.zeros((6, 6, 4))
    q_table = np.zeros((6, 6, 4))

    temp_a1_pos = 0
    temp_a2_pos = 0
    collision_action1_seq = []
    collision_action2_seq = []
    target1 = None
    target2 = None
    id = 0
    if valid_seq.count(False) != 0:
        collistion_index = valid_seq.index(False)
        # print(collistion_index)
        # print(len(agent1_action_sequence))
        if collistion_index == 0:
            temp_a1_pos = agent1_pos
            temp_a2_pos = agent2_pos
        else:
            temp_a1_pos = a1_seq[collistion_index-1][-1]
            temp_a2_pos = a2_seq[collistion_index-1][-1]
        if collistion_index % 2 == 0:
            id = 0
        else:
            id = 1
        collision_action1_seq = agent1_action_sequence[int(
            collistion_index/2)][id]
        collision_action2_seq = agnet2_action_sequence[int(
            collistion_index/2)][id]
        target1 = a1_seq[collistion_index][-1]
        target2 = a2_seq[collistion_index][-1]
        print(temp_a1_pos)
        print(temp_a2_pos)
        print(collision_action1_seq)
        print(collision_action2_seq)
        print(target1)
        print(target2)
        simulate(temp_a1_pos, temp_a2_pos,
                 collision_action1_seq, collision_action2_seq, target1, target2)

    # collision_index = valid_seq.index(False)
    # temp_agent1_location =
    # simulate()
