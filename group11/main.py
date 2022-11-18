import sys
import numpy as np
import math
import random
import gym
import warehouse_grid


def simulate():
    global epsilon, epsilon_decay
    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset()
        total_reward = 0
        best_action_sequence = []

        i = 0
        # AI tries up to MAX_TRY times
        for t in range(MAX_TRY):

            # In the beginning, do random action to learn
            agent1 = state[0]
            agent2 = state[1]
            num_action = 0
            if ac1 > ac2:
                num_action = ac2
            else:
                num_action = ac1

            new_state = None
            reward = 0
            done = False

            action1 = 10
            action2 = 10
            ## agent1 moving ##

            # for i in range(num_action):

            action = env.action_space.sample()
            print(x_agent, y_agent)
            if random.uniform(0, 1) < epsilon and y_agent == 5:
                ls = [0, 1, 2, 3, 5]
                action = random.choice(ls)
            elif random.uniform(0, 1) < epsilon and y_agent < 5:
                ls = [0, 1, 2, 3, 4]
                action = random.choice(ls)
            else:
                if y_agent == 5:
                    ls = q_table[x_agent, y_agent]
                    ls[4] = min(ls)
                    action = np.argmax(ls)
                elif y_agent < 5:
                    ls = q_table[x_agent, y_agent]
                    ls[5] = min(ls)
                    action = np.argmax(ls)

            # Do action and get result
            next_state, reward, done, _, = env.step(action)
            total_reward += reward

            # Get correspond q value from state, action pair
            next_agent_x = next_state[0]
            next_agent_y = next_state[1]
            q_value = q_table[x_agent, y_agent][action]
            best_q = np.max(q_table[next_agent_x, next_agent_y])

            # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
            q_table[x_agent, y_agent][action] = (
                1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

            # Set up for the next iteration
            state = next_state

            # Draw games
            env.render()

            # When episode is done, print reward
            if done or t >= MAX_TRY - 1:
                print("Episode %d finished after %i time steps with total reward = %f." % (
                    episode, t, total_reward))
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
            val_seq.append(valid)
            pos1 = a1_pos[len(a1_pos)-1]
            pos2 = a2_pos[len(a2_pos)-1]
            # print(valid)

    return val_seq


if __name__ == "__main__":
    env = gym.make("warehouse_grid/GridWorld-v0")
    MAX_EPISODES = 200
    MAX_TRY = 100
    epsilon = 0.9
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6
    agent1_action_sequence, agnet2_action_sequence = env.get_action_sequence()
    print(agent1_action_sequence)
    print(agnet2_action_sequence)
    agent1_pos, agent2_pos = env.get_agents_location()
    # print(validation(agent1_pos, agent2_pos,
    #       agent1_action_sequence[0][1], agnet2_action_sequence[0][1]))
    print(valid_concatenate(agent1_pos, agent2_pos,
          agent1_action_sequence, agnet2_action_sequence))
    agent1_q_table = np.zeros((5, 5, 7))
    agent2_q_table = np.zeros((5, 5, 7))
    q_table = np.zeros((5, 5, 6))
    # simulate()
