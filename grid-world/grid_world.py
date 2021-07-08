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
from matplotlib.table import Table
import os


class GridWorld:

    A_PRIME_POS = [4, 1]
    B_PRIME_POS = [2, 3]

    # left, up, right, down
    ACTIONS = [np.array([0, -1]),
               np.array([-1, 0]),
               np.array([0, 1]),
               np.array([1, 0])]
    ACTION_PROB = 0.25

    def __init__(self,DISCOUNT = 0.9,WORLD_SIZE = 5, A_POS = [0,1],B_POS = [0,3], showAllGraph = False, outProb = 0.25):
        self.WORLD_SIZE = WORLD_SIZE
        self.DISCOUNT = DISCOUNT
        self.A_POS = A_POS
        self.B_POS = B_POS
        self.showAllGraph = showAllGraph
        self.outProb = outProb



    def step(self,state, action):
        if state == self.A_POS:
            return self.A_PRIME_POS, 10, self.ACTION_PROB
        if state == self.B_POS:
            return self.B_PRIME_POS, 5, self.ACTION_PROB

        state = np.array(state)
        xCur,yCur = state.tolist()
        next_state = (state + action).tolist()
        x, y = next_state
        if x < 0 or x >= self.WORLD_SIZE or y < 0 or y >= self.WORLD_SIZE:
            reward = -1.0
            next_state = state
            prob = self.outProb
        else:
            if(xCur == 0 or yCur == 0 or xCur == self.WORLD_SIZE or yCur == self.WORLD_SIZE):
                prob = (1-self.outProb)/3
            reward = 0
            prob = self.ACTION_PROB
        return next_state, reward,prob

    def draw_image(self,image):
        fig, ax = plt.subplots()
        ax.set_axis_off()
        tb = Table(ax, bbox=[0, 0, 1, 1])

        nrows, ncols = image.shape
        width, height = 1.0 / ncols, 1.0 / nrows

        # Add cells
        for (i,j), val in np.ndenumerate(image):
            # Index either the first or second item of bkg_colors based on
            # a checker board pattern
            idx = [j % 2, (j + 1) % 2][i % 2]
            color = 'white'

            tb.add_cell(i, j, width, height, text=val,
                        loc='center', facecolor=color)

        # Row Labels...
        for i, label in enumerate(range(len(image))):
            tb.add_cell(i, -1, width, height, text=label+1, loc='right',
                        edgecolor='none', facecolor='none')
        # Column Labels...
        for j, label in enumerate(range(len(image))):
            tb.add_cell(-1, j, width, height/2, text=label+1, loc='center',
                               edgecolor='none', facecolor='none')
        ax.add_table(tb)

    def ValueUpdateBellman(self):
        k = 0
        value = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
        while True:
            k = k + 1
            # keep iteration until convergence
            new_value = np.zeros(value.shape)
            for i in range(0, self.WORLD_SIZE):
                for j in range(0,self.WORLD_SIZE):
                    for action in self.ACTIONS:
                        (next_i, next_j), reward, prob = self.step([i, j], action)
                        # bellman equation
                        new_value[i, j] += prob * (reward + self.DISCOUNT * value[next_i, next_j])

            if(self.showAllGraph):
                self.draw_image(np.round(new_value, decimals=2))
                filename = 'figure_1_' + str(k)+'.png'
                path = os.path.join('.', 'images', filename)
                plt.savefig(path)
                plt.show()

            if np.sum(np.abs(value - new_value)) < 1e-1:
                self.draw_image(np.round(new_value, decimals=2))
                plt.show()
                plt.close()
                print('Value function converges after ' + str(k) + ' loops')
                break

            value = new_value

    def OptimalValueUpdate(self):
        value = np.zeros((self.WORLD_SIZE, self.WORLD_SIZE))
        k = 0
        while True:
            # keep iteration until convergence
            k = k + 1
            new_value = np.zeros(value.shape)
            for i in range(0, self.WORLD_SIZE):
                for j in range(0, self.WORLD_SIZE):

                    values = []
                    for action in self.ACTIONS:
                        (next_i, next_j), reward, prob = self.step([i, j], action)
                        # value iteration
                        values.append(reward + self.DISCOUNT * value[next_i, next_j])
                    new_value[i, j] = np.max(values)
            if(self.showAllGraph):
                self.draw_image(np.round(new_value, decimals=2))
                filename = 'figure_2_' + str(k)+'.png'
                path = os.path.join('.', 'images', filename)
                plt.savefig(path)
                plt.show()
            if np.sum(np.abs(new_value - value)) < 1e-1:
                self.draw_image(np.round(new_value, decimals=2))
                plt.show()
                plt.close()
                print('Optimal value function is obtained after ' + str(k) + ' loops')

                break
            value = new_value

if __name__ == '__main__':
    game = GridWorld(showAllGraph = True)
    game.ValueUpdateBellman()
#    game.OptimalValueUpdate()
    print("success")

