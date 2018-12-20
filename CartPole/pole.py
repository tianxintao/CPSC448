# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gym
import numpy as np
from sklearn.neighbors import NearestNeighbors



env = gym.make('CartPole-v0')


class Agent:
    
    batchSize = 10000
    tupleSum = 0
    epsilon = 0.05;
    tuples = []
    value = []
    learningRate = 0.3
    discount = 0.9
    
    def _init_(self,randomLimit = 1000):
        self.randomLimit = randomLimit
        
    def generatingTuples(self):
        for i_episode in range(100):
            observation = env.reset()
            for t in range(10000000000):
                if (t >= 1):
                    previousObservation = observation
                action = self.TakeAction(observation)
                observation, reward, done, info = env.step(action)
                currentState = np.concatenate((previousObservation,action,reward,observation),axis = 0)
                self.tuples = np.concatenate((self.tuples,currentState),axis = 1)
                print("Observation is %s, reward is %s, action is %s" %(observation,reward,action))
                if done:
                    self.tupleSum += (t+1)
                    break
                
    # action is -1 or 1, if tuple size is less than randomLimit, take random action
    # Otherwise, take the action computed by KNN
    def TakeAction(self,observation):
        if(self.tupleSum <= self.randomLimit):
            return env.action_space.sample()
        else:
            rightForceValue = self.KNNApprox(observation,1)
            leftForceValue = self.KNNApprox(observation,-1)
            
            if(rightForceValue > leftForceValue):
                return 1
            else:
                return -1
            
    # Q learning to compute the optimal control policy
    def ValueIteration(self):
        self.value = np.zeros((self.tuples.shape(0), 6))
        while True:
            new_value = np.zeros(self.value.shape)
            for ind in range(self.tuples.shape(0)):
                tdError = self.tuples[ind,5]+self.discount*max(self.KNNApprox(self.tuples[ind,6:9],-1),self.KNNApprox(self.tuples[ind,6:9],1))-self.value[ind,5]
                new_value[ind,5] = self.value[ind,5] + self.learningRate*(tdError)
            
            
            if np.sum(np.abs(self.value - new_value)) < 1e-1:
                print('Value function converges')
                break            
                
                
    def KNNApprox(self,state,action):
        train = self.value[:,0:4]
        nbrs = NearestNeighbors(n_neighbors=3).fit(train)
        _,indices = nbrs.kneighbors(np.concatenate(state,action,axis = 0))
        
        return np.mean(self.value[indices,5]);
            
            
        