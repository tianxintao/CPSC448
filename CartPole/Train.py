#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 22:16:21 2018

@author: tianxin
"""

"""
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
    Observation: 
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                 -24°           24°
        3	Pole Velocity At Tip      -Inf            Inf
        
    Actions:
        Type: Discrete(2)
        Num	Action
        0	Push cart to the left
        1	Push cart to the right
        
        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value between ±0.05
    Episode Termination:
        Pole Angle is more than ±12°
        Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
        Episode length is greater than 200
        Solved Requirements
        Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
    """
    
    
from pole import Agent
import numpy as np
import gym


agent = Agent(randomLimit= 100)




def FindBestK():
    KArr = range(1,10)
#    KArr = np.concatenate((range(10),range(10,100,10),range(100,1000,100)),axis = 0)
    SumRounds = np.zeros(len(KArr))
    for kNum in KArr:
        agent = Agent(randomLimit = 100, k = kNum)
        print(agent.value,agent.tuples)
        SumRounds[kNum-1] = agent.PlayGameNTimes(1)
        print("One K is computed")
    print("The best K for KNN algorithm is %s" % (KArr[np.argmax(SumRounds)]))
    
def FindBestDistanceWeights():
    bestScore = 0
    weight = np.zeros(5)
    bestWeight = np.zeros(5)
    
    interval = 5
    ratio = 1.0/interval
    
    for i in range(interval):
        weight[4] = i*ratio
        for j in range(interval):
            weight[0] = (j*ratio)/2
            weight[2] = (j*ratio)/2
            weight[1] = (1-j*ratio)/2
            weight[3] = (1-j*ratio)/2
            agent = Agent(randomLimit=50,weight=weight)
            score = agent.PlayGameNTimes(9)
            if(score > bestScore):
                bestScore = score
                bestWeight = weight
    
    print("The best weight is:")
    print(bestWeight)
        
        
        
        
if __name__ == '__main__':
    agent = Agent(randomLimit = 100)
    agent.LoadPolicy()
    agent.tupleSum = 8411