# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gym
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing


env = gym.make('CartPole-v1')


class Agent:
    
    batchSize = 10000
    tupleSum = 0
    epsilon = 0.1;

    learningRate = 0.05
    discount = 0.7
    tupleSum = 0
    
    def __init__(self,randomLimit = 1000,k = 6,weight = [1,1,1,1,1]):
        self.randomLimit = randomLimit
        self.k = k
        self.tuples = np.zeros((100000,10))
        self.value = np.zeros((100000,6))
        self.weight = weight
    
    
    
    def mydist(self,x,y):
        return np.sum(np.dot((x-y)**2,self.weight))
    
    def generatingTuples(self):
        rounds = 0
        for i_episode in range(2):
            previousObservation = env.reset()
            for t in range(5000):
                if (t >= 1):
                    previousObservation = observation
                action = self.TakeAction(previousObservation)
                observation, reward, done, info = env.step(action)
                currentTuple = np.concatenate((previousObservation,[action],[reward],observation),axis = 0)
                currentState = np.concatenate((previousObservation,[action],[0]),axis = 0)
                self.tuples[self.tupleSum] = currentTuple
                self.value[self.tupleSum] = currentState
                self.tupleSum = self.tupleSum + 1
#                print("Observation is %s, reward is %s, action is %s" %(observation,reward,action))
                if done:
                    print("game is over in %s rounds" %(t+1))
                    rounds = rounds + (t+1)
                    for k in range(2):
                        currentTuple = np.concatenate((observation,[k],[reward],[999,999,999,999]),axis = 0)
                        currentState = np.concatenate((observation,[k],[-1]),axis = 0)
                        self.tuples[self.tupleSum] = currentTuple
                        self.value[self.tupleSum] = currentState
                        self.tupleSum = self.tupleSum + 1
                    break
        train = self.value[0:self.tupleSum,0:5]
        self.scaler = preprocessing.StandardScaler().fit(train)
        train = self.scaler.transform(train)
        self.nbrs = NearestNeighbors(n_neighbors=self.k,metric=self.mydist).fit(train)
        return rounds
    
                
    # action is -1 or 1, if tuple size is less than randomLimit, take random action
    # Otherwise, take the action computed by KNN
    def TakeAction(self,observation):
        if(self.tupleSum <= self.randomLimit) or np.random.binomial(1, self.epsilon) == 1:
            return env.action_space.sample()
        else:
            rightForceValue = self.KNNApprox(observation,1)
            leftForceValue = self.KNNApprox(observation,-1)
            
            if(rightForceValue > leftForceValue):
                return 1
            else:
                return 0
            
    # Q learning to compute the optimal control policy
    def ValueIteration(self):
        count = 0
        print("Value Iteration")
        while True:
            if count >=1:
                self.value = self.new_value;
            self.new_value = np.zeros(self.value.shape)
            self.new_value[:,0:5] = self.value[:,0:5]
            for ind in range(self.tupleSum):
                count += 1
#                print(count)
                if(self.tuples[ind,6]!= 999):
                    tdError = self.tuples[ind,5]+self.discount*max(self.KNNApprox(self.tuples[ind,6:10],0),self.KNNApprox(self.tuples[ind,6:10],1))-self.value[ind,5]
                    self.new_value[ind,5] = self.value[ind,5] + self.learningRate*(tdError)
#                    print(self.new_value[ind,5])
#            if count >= 150000:
#                self.value = self.new_value
#                break;
            if np.sum(np.abs(self.value - self.new_value)) < 1e-1:
                print('Value function converges')
                self.value = self.new_value
                break            
                
                
    def KNNApprox(self,state,action):
#        print(np.concatenate((state,[action])))
        temp,indices = self.nbrs.kneighbors(self.scaler.transform(np.concatenate((state,[action]),axis = 0).reshape(1,5)))
#        print(np.mean(self.value[indices,5]))
        return np.mean(self.value[indices,5]);
    
    
    
    def PlayGameNTimes(self,n):
        totalRounds = 0
        for j in range(n):
            rounds = self.generatingTuples()
            if j <= 15:
                self.ValueIteration()
            totalRounds = totalRounds + rounds
        
        return totalRounds

            
    
    

if __name__ == '__main__':
    agent = Agent()
    agent.generatingTuples()

    tuples = agent.tuples
    agent.ValueIteration()
    values = agent.value

            
        