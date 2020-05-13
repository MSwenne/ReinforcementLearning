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
import concurrent.futures
from hex_skeleton import HexBoard
from operator import itemgetter 
import numpy as np
import random
import heapq
import time
import copy
from utils import copy_board

def gen_random(low, high):
    return np.random.randint(low=0, high=high, size=1)[0]

class MCTS:
    def __init__(self, Cp, itermax, max_time):
        self.Cp = Cp
        self.itermax = itermax
        self.max_time = max_time

    def makeMove(self, board, color):
        self.maximizing_color = color
        roots = []
        for _ in range(100):
            newBoard = copy_board(board)
            cp = self.Cp
            roots.append((newBoard, cp))
        # For each of the tmp_roots we run in parallel the mcts process.
        with concurrent.futures.ProcessPoolExecutor() as executor: 
            results = [executor.submit(self.performMCTS, p_board, color, cp) for (p_board, cp) in roots]
            all_searched = []
            for f in concurrent.futures.as_completed(results):
                all_searched.append(f.result())
            winner = max(all_searched, key = itemgetter(0))[1]
            return winner.getMove()

    def performMCTS(self, board, color, cp):
    
        root = Node(board, None, color, None)
        it = 0
        curr_time = time.time()
        while it < self.itermax and time.time() - curr_time < self.max_time:
            # Selection 
            curr = self.selectPromising(root, cp, it)
            
            # Expansion
            if curr.getBoard().is_game_over():
                child = curr
            else:
                child = self.expand(curr)
            # Simulation
            result = self.playout(child)
            # Backpropagation
            while child != None:
                child.updateState(result)
                child = child.getParent()
            it += 1
        winner = self.getMostVisited(root)
        self.delete(root)
        return winner

    def getMostVisited(self, root):
        UCTs = [(child.state[1],child) for child in root.getChildren()]
        return max(UCTs, key = itemgetter(0))

    def selectPromising(self, root, cp, it):
        curr = root
        while len(curr.getBoard().getMoveList(curr.getColor())) ==  len(curr.getChildren()):
            if curr.getBoard().is_game_over():
                return curr
            curr = self.findBestUCT(curr, cp)
        return curr


    def findBestUCT(self, curr, cp):
        UCTs = [(self.UCT(child, cp),child) for child in curr.getChildren()]
        return max(UCTs, key = itemgetter(0))[1]
        

    def UCT(self, curr, cp):
        win, visit = curr.getState()
        _, parent_visit = curr.getParent().getState()
        return (win / visit) + 2 * cp * np.sqrt(2* np.log(parent_visit) / visit)

    def expand(self,curr):
        board = copy_board(curr.getBoard())
        color = curr.getColor()
        if(len(curr.getMoves()) != 0):
            move = curr.ExpandRandomMove()
            board.place(move, color)
            child = Node(board, move, board.get_opposite_color(color), curr)
            curr.addChild(child)
            return child
        return curr

    def playout(self, curr):
        board = curr.getBoard()
        player = curr.getColor()
        win, color = self.recursivePlayout(copy_board(board), player)
        if win and color == self.maximizing_color:
            return 1
        return -1

    def recursivePlayout(self, board, color):
        if board.is_game_over():
            return board.check_win(board.get_opposite_color(color)), board.get_opposite_color(color)
        moves = board.getMoveList(color)
        move = moves[random.randint(0,len(moves)-1)]
        board.place(move, color)
        return self.recursivePlayout(board, board.get_opposite_color(color))

    def delete(self, root):
        for child in root.getChildren():
            self.delete(child)
        del root

class Node:
    def __init__(self, board, move, color, parent):
        self.board = board
        self.move = move
        self.moves = self.board.getMoveList(color)
        self.color = color
        self.state = (0,0) # (win, visit)
        self.parent = parent
        self.children = []

    def getBoard(self):
        return self.board

    def getMove(self):
        return self.move

    def getMoves(self):
        return self.moves

    def ExpandRandomMove(self):
        if len(self.moves) == 0:
            self.getBoard().print()
            return None
        return self.moves.pop(gen_random(0, len(self.moves)))

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