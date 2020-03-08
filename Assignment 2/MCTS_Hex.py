#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 2: HEX                          #
#   MCTS class                                                      #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 6 March 2020                                         #
# All rights reserved                                               #
#                                                                   #
#####################################################################

from hex_skeleton import HexBoard
import numpy as np
import random
import heapq
import time
MAX_TIME = 0.1

class MCTS:
    def __init__(self, Cp):
        self.Cp = Cp
        self.visit = 0

    def makeMCTSmove(self):
        pass

    def MCTS(self, board, color, itermax):
        root = Node(board, color, None)
        curr = time.time()
        while time.time() - curr < MAX_TIME:
            curr = self.selectPromising(root)
            if not curr.getBoard().is_game_over():
                self.expand(curr)
            explore = curr
            if len(curr.getChildren()) > 0:
                explore = self.randomMove(curr)
            result = self.simulate(explore)
            self.backPropogation(explore, result)
        winner = self.findBestUCT(root)
        return winner.getBoard()

    def selectPromising(self, root):
        curr = root
        while len(curr.getChildren()) == len(curr.getChildren()):
            curr = self.findBestUCT(curr)
        return curr

    def findBestUCT(self, curr):
        if len(curr.getBoard().getMoveList()) == len(curr.getChildren()):
            UCTs = [(self.UCT(child),child) for child in curr.getChildren()]
            return max(UCTs)[1]

    def UCT(self, curr):
        win, visit = curr.getState()
        return win / visit - self.Cp * np.log(self.visit / visit)

    def expand(self,curr):
        board = curr.getBoard()
        color = curr.getColor()
        board = self.randomMove(board, color)
        child = Node(board, board.get_opposite_color(color), curr)
        curr.addChild(child)
        result = self.playout(child)
        while child.getParent() != None:
            child.updateState(result)
            child = child.getParent()
        child.updateState(result)

    def playout(self, curr):
        board = curr.getBoard()
        player = curr.getColor()
        color = player
        while not board.is_game_over():
            board = self.randomMove(board, color)
            color = board.get_opposite_color(color)
        return int(board.check_win(player))

    def randomMove(self, board,color):
        moves = board.getMoves(color)
        board.place(moves[random.randint(0,len(moves))])
        return board

class Node:
    def __init__(self, board, color, parent):
        self.board = board
        self.color = color
        self.state = (0,0) # (win, visit)
        self.parent = parent
        self.children = []

    def getBoard(self):
        return self.board

    def getColor(self):
        return self.color

    def getState(self):
        return self.state[0], self.state[1]

    def setState(self, win, visit):
        self.state = (win, visit)

    def updateState(self, result):
        self.setState(self.state[0]+result, self.state[1]+1)

    def getParent(self):
        return self.parent

    def getChildren(self):
        return self.children

    def addChild(self, child):
        self.children.append(child)