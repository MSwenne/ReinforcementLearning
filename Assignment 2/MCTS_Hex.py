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


class MCTS:
    def makeMCTSmove():
        pass

    def MCTS(rootstate, itermax):
        curr = time.time()
        while time.time() - curr < MAX_TIME:
            print("Hello")
        rootnode = Node(state=rootstate)
        for i in range(itermax):
            node = rootnode
            state = rootstate.Clone()

            while node.untriedMove == [] and node.childNodes != []:
                node = node.UCTSelectChild()
                state.DoMove(node.move)
            if node.untriedMoves != []:
                m = random.choice(node.untriedMove)
                state.DoMove(m)
                node = node.AddChild(m, state)

            while state.getMoves() != []:
                state.DoMove(random.choice(state.getMoves()))

            while node != None:
                node.Update(state.GetResult(node.playerJustMoved))
                node = node.parentNode

        return sorted(rootnode.childNodes, key= lambda c:  c.visits)[-1].move
