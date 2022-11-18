import numpy as np


def cost_distance(current_position, target_position):
    return abs(sum(current_position) - sum(target_position))


# Goal => [[basekt_position, item],[basket_position, item], ... ]
def create_cost_table(agent_position, goal_list):
    number_of_agent = 2
    cost_table = np.zeros((2, len(goal_list)))
    for r in range(2):
        for i, g in enumerate(goal_list):
            cost = cost_distance(agent_position[r], g[1])
            cost_table[r][i]
    return cost_table


def split_goal(agent_position, baskets):
    agent1_goals = []
    agent2_goals = []
    goal_list = []
    for i in baskets:
        goal_list.append([i.pos, i.item])
    cost_table = create_cost_table(agent_position, goal_list)

    while len(goal_list) > 1:
        agent_1_first_goal = np.argmin(cost_table[0])
        agent_2_first_goal = np.argmin(cost_table[1])

        if agent_1_first_goal == agent_2_first_goal:
            if len(agent1_goals) < len(agent2_goals):
                agent1_goals.append(goal_list[agent_1_first_goal])
                cost_table = np.delete(cost_table, agent_1_first_goal, 1)
                agent_2_first_goal = np.argmin(cost_table[1])
                agent2_goals.append(goal_list[agent_2_first_goal])
                cost_table = np.delete(cost_table, agent_2_first_goal, 1)
                goal_list.pop(agent_1_first_goal)
                goal_list.pop(agent_2_first_goal)
            else:
                agent2_goals.append(goal_list[agent_2_first_goal])
                cost_table = np.delete(cost_table, agent_2_first_goal, 1)
                agent_1_first_goal = np.argmin(cost_table[0])
                agent1_goals.append(goal_list[agent_1_first_goal])
                goal_list.pop(agent_1_first_goal)
                goal_list.pop(agent_2_first_goal)
        elif agent_1_first_goal > agent_2_first_goal:
            agent1_goals.append(goal_list[agent_1_first_goal])
            agent2_goals.append(goal_list[agent_2_first_goal])
            goal_list.pop(agent_1_first_goal)
            goal_list.pop(agent_2_first_goal)
        elif agent_1_first_goal < agent_2_first_goal:
            agent2_goals.append(goal_list[agent_2_first_goal])
            agent1_goals.append(goal_list[agent_1_first_goal])
            goal_list.pop(agent_2_first_goal)
            goal_list.pop(agent_1_first_goal)
        current_agent1_position = agent_1_first_goal[0]
        current_agent2_position = agent_2_first_goal[0]
        cost_table = create_cost_table(
            [current_agent1_position, current_agent2_position], goal_list)
    return agent1_goals, agent2_goals
