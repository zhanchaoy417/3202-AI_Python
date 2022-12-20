import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from collections import deque
import heapq
import unittest

class Environment:
    def __init__(self, size: tuple = (10, 10), seed: int = 0):
        np.random.seed(seed)
        self.size = size
        self.seed = seed
        self.env = self.__create_env()
        # need to set these in particular with special functions that they need to solve
        self.edge_weights_1 = {}
        self.edge_weights_2 = {}
        self.__set_edge_weights()

    def show_env_and_path(self, path):
        env_copy = np.copy(self.env)

        if(len(path) < 1):
            # if your path solver functions should return [] or [], 0 when no solution is found
            print("Empty path given to show_env_and_path()")
            return
        
        for cell in path:
            if(env_copy[cell[0]][cell[1]] >= 5 and env_copy[cell[0]][cell[1]] < 20):
                env_copy[cell[0]][cell[1]] = 41


        # color start node
        start = path[0]
        goal = path[-1]

        # start node
        env_copy[start[0]][start[1]] = 35        
        
        # goal node
        env_copy[goal[0]][goal[1]] = 25


        self.show_env(env_copy)


    def show_env(self, new_env):
        #if(env == None):
        #    env = self.env

        cmap = colors.ListedColormap(['yellow', 'teal', 'pink', 'grey', 'red'])
        bounds = [0,5,20,30,40,41]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(new_env, cmap=cmap, norm=norm)
        ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
        ax.set_xticks(np.arange(-.5, 10, 1))
        ax.set_yticks(np.arange(-.5, 10, 1))
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])

        plt.show()
    def __create_env(self):
        env = np.random.rand(self.size[0], self.size[1]) * 20
        return env


    def __set_edge_weights(self):
        # create dictionary

        for row, row_vals in enumerate(self.env):
            for col, val in enumerate(self.env[row]):
                # create dictionary
                self.edge_weights_1[(row, col)] = {}
                self.edge_weights_2[(row, col)] = {}

                #set all 6 direction options in edge_wights
                # 1) up
                if(row > 0):
                    self.edge_weights_1[(row, col)][(row-1, col)] = 1
                    self.edge_weights_2[(row, col)][(row-1, col)] = np.random.randint(101)
                    if(col > 0):
                        self.edge_weights_1[(row, col)][(row-1, col-1)] = 1
                        self.edge_weights_2[(row, col)][(row-1, col-1)] = np.random.randint(101)
                    if(col < 9):
                        self.edge_weights_1[(row, col)][(row-1, col+1)] = 1
                        self.edge_weights_2[(row, col)][(row-1, col+1)] = np.random.randint(101)


                #2) left
                if(col > 0):
                    self.edge_weights_1[(row, col)][(row, col-1)] = 1
                    self.edge_weights_2[(row, col)][(row, col-1)] = np.random.randint(101)
                    if(row < 9):
                        self.edge_weights_1[(row, col)][(row+1, col-1)] = 1
                        self.edge_weights_2[(row, col)][(row+1, col-1)] = np.random.randint(101)

                #3) down
                if(row < 9):
                        self.edge_weights_1[(row, col)][(row+1, col)] = 1
                        self.edge_weights_2[(row, col)][(row+1, col)] = np.random.randint(101)
                        if(col < 9):
                            self.edge_weights_1[(row, col)][(row+1, col+1)] = 1
                            self.edge_weights_2[(row, col)][(row+1, col+1)] = np.random.randint(101)

                #) right
                if(col < 9):
                        self.edge_weights_1[(row, col)][(row, col+1)] = 1
                        self.edge_weights_2[(row, col)][(row, col+1)] = np.random.randint(101)

        for first_node in list(self.edge_weights_2.keys()):
            for second_node in list(self.edge_weights_2[first_node].keys()):

                # if first_node is yellow (an obstacle) the connection going both ways
                if(self.env[first_node[0]][[first_node[1]]] < 5):
                    self.edge_weights_2[first_node].pop(second_node)
                    self.edge_weights_2[second_node].pop(first_node)
                    self.edge_weights_1[first_node].pop(second_node)
                    self.edge_weights_1[second_node].pop(first_node)
                # if there is a connection, make sure both edges are the same
                else:
                    w1 = self.edge_weights_2[first_node][second_node]
                    w2 = self.edge_weights_2[second_node][first_node]
                    if(w1 != w2):
                        self.edge_weights_2[first_node][second_node] = self.edge_weights_2[second_node][first_node]