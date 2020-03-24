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
    def __init__(self, Cp, itermax):
        self.Cp = Cp
        self.itermax = itermax

    def makeMove(self, board, color):
        root = Node(board, color, None)
        curr = time.time()
        for _ in range(self.itermax):
            # Selection
            curr = self.selectPromising(root)
            # Expansion
            if curr.getBoard().is_game_over():
                child = curr
            else:
                child = self.expand(curr)
            # Simulation
            result = self.playout(child)
            # Backpropagation
            while child.getParent() != None:
                child.updateState(result)
                child = child.getParent()
            child.updateState(result)
        winner = self.findBestUCT(root, max)
        winner = winner.getBoard()
        self.delete(root)
        return winner

    def selectPromising(self, root):
        curr = root
        minimax = [min,max]
        maximize = True
        while len(curr.getBoard().getMoveList(curr.getColor())) == len(curr.getChildren()):
            if curr.getBoard().is_game_over():
                return curr
            curr = self.findBestUCT(curr,minimax[int(maximize)])
            maximize = not maximize
        return curr

    def findBestUCT(self, curr, minimax):
        UCTs = [(self.UCT(child),child) for child in curr.getChildren()]
        # states = [(self.UCT(child),child.getState()) for child in curr.getChildren()]
        return minimax(UCTs, key = itemgetter(0))[1]

    def UCT(self, curr):
        win, visit = curr.getState()
        _, parent_visit = curr.getParent().getState()
        return win / visit + self.Cp * np.sqrt(np.log(parent_visit) / visit)

    def expand(self,curr):
        board = copy.deepcopy(curr.getBoard())
        color = curr.getColor()
        if(len(curr.getMoves()) != 0):
            board.place(curr.ExpandRandomMove(), color)
            child = Node(board, board.get_opposite_color(color), curr)
            curr.addChild(child)
            return child
        return curr

    def playout(self, curr):
        board = curr.getBoard()
        player = curr.getColor()
        win, color = self.recursivePlayout(board, player)
        return int((win and color == player) or ((not win) and (color != player)))

    def recursivePlayout(self, board, color):
        if board.is_game_over():
            return board.check_win(color), color
        moves = board.getMoveList(color)
        move = moves[random.randint(0,len(moves)-1)]
        board.place(move, color)
        win, color = self.recursivePlayout(board, board.get_opposite_color(color))
        board.unplace(move)
        return win, color

    def delete(self, root):
        for child in root.getChildren():
            self.delete(child)
        del root

class Node:
    def __init__(self, board, color, parent):
        self.board = board
        self.moves = self.board.getMoveList(color)
        self.color = color
        self.state = (0,0) # (win, visit)
        self.parent = parent
        self.children = []

    def getBoard(self):
        return self.board

    def getMoves(self):
        return self.moves

    def ExpandRandomMove(self):
        if len(self.moves) == 0:
            self.getBoard().print()
            return None
        return self.moves.pop(random.randint(0,len(self.moves)-1))

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