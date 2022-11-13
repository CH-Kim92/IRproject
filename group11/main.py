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
        state, _ = env.reset()
        total_reward = 0

        # AI tries up to MAX_TRY times
        for t in range(MAX_TRY):

            # In the beginning, do random action to learn
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                x_agent = state['agent'][0]
                y_agent = state['agent'][1]
                action = np.argmax(q_table[x_agent, y_agent])

            # Do action and get result
            next_state, reward, done, _, info = env.step(action)
            total_reward += reward

            # Get correspond q value from state, action pair
            x_agent = state['agent'][0]
            y_agent = state['agent'][1]
            next_agent_x = next_state['agent'][0]
            next_agent_y = next_state['agent'][1]
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


if __name__ == "__main__":
    env = gym.make("warehouse_grid/GridWorld-v0")
    MAX_EPISODES = 4000
    MAX_TRY = 100
    epsilon = 0.9
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6
    num_box = tuple((env.observation_space["agent"].high +
                    np.ones(env.observation_space.shape)).astype(int))
    q_table = np.zeros(num_box + (env.action_space.n,))
    simulate()
    print(q_table)
