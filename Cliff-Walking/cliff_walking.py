#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm





class CliffWalking:
    # world height
    WORLD_HEIGHT = 4
    
    # world width
    WORLD_WIDTH = 12
    
    # probability for exploration
    EPSILON = 0.1
    
    # all possible actions
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
    
    # initial state action pair values
    START = [3, 0]
    GOAL = [3, 11]
    
    def __init__(self,penalty = -100):
        self.penalty = penalty
    
    def step(self,state, action):
        i, j = state
        if action == self.ACTION_UP:
            next_state = [max(i - 1, 0), j]
        elif action == self.ACTION_LEFT:
            next_state = [i, max(j - 1, 0)]
        elif action == self.ACTION_RIGHT:
            next_state = [i, min(j + 1, self.WORLD_WIDTH - 1)]
        elif action == self.ACTION_DOWN:
            next_state = [min(i + 1, self.WORLD_HEIGHT - 1), j]
        else:
            assert False
    
        reward = -1
        
        if (action == self.ACTION_DOWN and i == 2 and 1 <= j <= 10) or (
            action == self.ACTION_RIGHT and state == self.START):
            reward = self.penalty
            next_state = self.START
    
    
        return next_state, reward
    
    # choose an action based on epsilon greedy algorithm
    def choose_action(self,state, q_value):
        if np.random.binomial(1, self.EPSILON) == 1:
            return np.random.choice(self.ACTIONS)
        else:
            ## values here means Q(s',a')
            values_ = q_value[state[0], state[1], :]
            return np.random.choice([action for action, value_ in enumerate(values_) if value_ == np.max(values_)])
    
    # an episode with Sarsa
    # @q_value: values for state action pair, will be updated
    # @expected: if True, will use expected Sarsa algorithm
    # @step_size: step size for updating
    # @return: total rewards within this episode
    def sarsa(self,q_value, expected=False, step_size = 0.9):
        state = self.START
        action = self.choose_action(state, q_value)
        rewards = 0.0
        while state != self.GOAL:
            next_state, reward = self.step(state, action)
            next_action = self.choose_action(next_state, q_value)
            rewards += reward
            if not expected:
                target = q_value[next_state[0], next_state[1], next_action]
            else:
                # calculate the expected value of new state
                target = 0.0
                q_next = q_value[next_state[0], next_state[1], :]
                best_actions = np.argwhere(q_next == np.max(q_next))
                for action_ in self.ACTIONS:
                    if action_ in best_actions:
                        target += ((1.0 - self.EPSILON) / len(best_actions) + self.EPSILON / len(self.ACTIONS)) * q_value[next_state[0], next_state[1], action_]
                    else:
                        target += self.EPSILON / len(self.ACTIONS) * q_value[next_state[0], next_state[1], action_]
            target *= 0.7
            q_value[state[0], state[1], action] += step_size * (
                    reward + target - q_value[state[0], state[1], action])
            state = next_state
            action = next_action
        return rewards
    
    # an episode with Q-Learning
    # @q_value: values for state action pair, will be updated
    # @step_size: step size for updating
    # @return: total rewards within this episode
    def q_learning(self,q_value,step_size = 0.9):
        state = self.START
        rewards = 0.0
        while state != self.GOAL:
            action = self.choose_action(state, q_value)
            next_state, reward = self.step(state, action)
            rewards += reward
            # Q-Learning update
            q_value[state[0], state[1], action] += step_size * (
                    reward + step_size * np.max(q_value[next_state[0], next_state[1], :]) -
                    q_value[state[0], state[1], action])
            state = next_state
        return rewards
    
    
    
    
    
    def double_q_learning(self,q_value_1,q_value_2, step_size=0.9):
        state = self.START
        rewards = 0.0
        while state != self.GOAL:
            action = self.choose_action(state, np.add(q_value_1,q_value_2))
            next_state, reward = self.step(state, action)
            rewards += reward
            if (np.random.binomial(1, 0.5) == 1):
                actionFromQ1 = np.argmax(q_value_1[next_state[0],next_state[1],:])
                q_value_1[state[0], state[1], action] += step_size * (
                        reward + (q_value_2[next_state[0], next_state[1], actionFromQ1]) -
                        q_value_1[state[0], state[1], action])
            else:
                actionFromQ2 = np.argmax(q_value_2[next_state[0],next_state[1],:])
                q_value_2[state[0], state[1], action] += step_size * (
                        reward + (q_value_1[next_state[0], next_state[1], actionFromQ2]) -
                        q_value_2[state[0], state[1], action])
            state = next_state
        return rewards
    
    
    # print optimal policy
    def print_optimal_policy(self,q_value):
        optimal_policy = []
        for i in range(0, self.WORLD_HEIGHT):
            optimal_policy.append([])
            for j in range(0, self.WORLD_WIDTH):
                if [i, j] == self.GOAL:
                    optimal_policy[-1].append('G')
                    continue
                bestAction = np.argmax(q_value[i, j, :])
                if bestAction == self.ACTION_UP:
                    optimal_policy[-1].append('U')
                elif bestAction == self.ACTION_DOWN:
                    optimal_policy[-1].append('D')
                elif bestAction == self.ACTION_LEFT:
                    optimal_policy[-1].append('L')
                elif bestAction == self.ACTION_RIGHT:
                    optimal_policy[-1].append('R')
        for row in optimal_policy:
            print(row)
    

    def TestSarsa(self,episodes = 10000,stepSize = 0.5):
        # episodes of each run
        rewards_sarsa = np.zeros(episodes)
        q_sarsa = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, 4))
        for i in range(0, episodes):
            rewards_sarsa[i] += self.sarsa(q_sarsa,step_size = stepSize)
                 
        # averaging over independt runs
#        rewards_sarsa /= runs    
        print('Sarsa Optimal Policy:')
        self.print_optimal_policy(q_sarsa)
        
        
    def TestExpectedSarsa(self,episodes = 5000,stepSize = 0.5):
        # episodes of each run
        rewards_sarsa = np.zeros(episodes)
        q_sarsa = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, 4))
        for i in range(0, episodes):
            rewards_sarsa[i] += self.sarsa(q_sarsa,expected = True,step_size = stepSize)
             
        # averaging over independt runs
#        rewards_sarsa /= runs    
        print('Expected Sarsa Optimal Policy:')
        self.print_optimal_policy(q_sarsa)
        
    def TestQLearning(self,episodes = 5000,stepSize = 0.9):
        rewards_q_learning = np.zeros(episodes)
        q_q_learning = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, 4))
        for i in range(0, episodes):
             rewards_q_learning[i] += self.q_learning(q_q_learning,step_size = stepSize)
#        rewards_q_learning /= runs
        
        print('Q-Learning Optimal Policy:')
        self.print_optimal_policy(q_q_learning)
        
        
        
    def TestDoubleQLearning(self,episodes = 30000,stepSize = 0.9):
        rewards_q_double_learning = np.zeros(episodes)
        q_double_1 = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, 4))
        q_double_2 = np.zeros((self.WORLD_HEIGHT, self.WORLD_WIDTH, 4))
        for i in range(0, episodes):
             rewards_q_double_learning += self.double_q_learning(q_double_1,q_double_2,step_size = stepSize)
#        rewards_q_double_learning /= runs
        
        print('Double-Q-Learning Optimal Policy:')
        self.print_optimal_policy(np.add(q_double_1,q_double_2))


if __name__ == '__main__':
#    figure_6_6()
    game = CliffWalking()
    game.TestSarsa()
    game.TestExpectedSarsa()
