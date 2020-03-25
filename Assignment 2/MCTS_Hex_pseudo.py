#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 2: HEX                          #
#   MCTS class                                                      #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 24 March 2020                                        #
# All rights reserved                                               #
#                                                                   #
#####################################################################

from hex_skeleton import HexBoard
from operator import itemgetter 
import numpy as np
import random
import heapq
import time
import copy

class MCTS:
    def __init__(self, Cp, itermax, max_time):
        self.Cp = Cp
        self.itermax = itermax
        self.max_time = max_time

    def makeMove(self, rootboard, color):
        root = Node(None, None, color, rootboard.getMoveList(color))
        self.maximizing_color = color
        curr_time = time.time()
        it = 0
        while it < self.itermax and time.time() - curr_time < self.max_time:
            curr = root
            board = copy.deepcopy(rootboard)
            # Selection
            while len(curr.untried) == 0 and len(curr.children) != 0:
                curr = self.findBestUCT(curr)
                board.place(curr.move, curr.color)
            # Expansion
            if len(curr.untried) != 0:
                move = curr.untried.pop(random.randint(0, len(curr.untried)-1))
                board.place(move, curr.color)
                curr = curr.addChild(move, board)
            # Simulation
            while not board.is_game_over():
                moves = board.getMoveList(color)
                board.place(moves[random.randint(0,len(moves)-1)], color)
                color = board.get_opposite_color(color)
            # Backpropagation
            while curr != None:
                curr.updateState(int(board.check_win(self.maximizing_color)))
                curr = curr.parent
            del board
            it += 1
        winner = self.findBestUCT(root)
        rootboard.place(winner.move, self.maximizing_color)
        return rootboard

    def findBestUCT(self, curr):
        UCTs = [(self.UCT(child),child) for child in curr.children]
        return max(UCTs, key = itemgetter(0))[1]

    def UCT(self, curr):
        win, visit = curr.state
        _, parent_visit = curr.parent.state
        return win / visit + self.Cp * np.sqrt(np.log(parent_visit) / visit)


class Node:
    def __init__(self, parent, move, color, untried):
        self.state = (0,0) # (win, visit)
        self.move = move
        self.color = color
        self.untried = untried
        self.parent = parent
        self.children = []

    def updateState(self, result):
        self.state = (self.state[0]+result, self.state[1]+1)

    def addChild(self, move, board):
        color = board.get_opposite_color(self.color)
        child = Node(self, move, color, board.getMoveList(color))
        self.children.append(child)
        return child