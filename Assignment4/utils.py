#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 2: HEX                          #
#   utility functions                                               #
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
import copy

def get_input(message, valid, ending="\n"):
    print(message, end=ending)
    result = input()
    while result not in valid:
        print("Invalid value!")
        print(message, end=ending)
        result = input()
    return result


def copy_board(board):
    new_board = HexBoard(board.size)
    new_board.set_board(board.board.copy(), bool(board.game_over))
    return new_board

