#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 2: HEX                          #
#   Play class                                                      #
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


class Play():
    def __init__(self, player2, player1=None):
        self.p1 = player1 # if None, human play
        self.p2 = player2

    # Allows the user to play a game of Hex versus a bot
    # The bot uses alpha-beta search and a random evaluation
    def play_game(self):
        #initialize the board size and player color
        board, player = self.init_params()
        turn = 0
        color = [HexBoard.RED, HexBoard.BLUE]

        # Run a game until it is over
        while(not board.is_game_over()):
            # If it is the turn of the player...
            if turn == player:
                print("make a move...")
                # ... let him make a move
                self.makeMove(board, color[turn])
            # If it is the turn of the bot...
            else:
                print("enemy's turn:")
                # Generate a move for the bot
                self.player1.makeMove(board, color[turn], 3)
            # Print the board after every move
            board.print()
            # Switch turns
            turn = int(not turn)
        # If the game is over, show who has won
        if board.check_win(color[0]):
            print("RED wins!")
        else:
            print("BLUE wins!")

    # Function that asks questions to the user about the game parameters
    def init_params(self):
        print("Hex game: how big is the board? (minimal 2x2 and maximal 10x10)")
        # Ask the user for a board size
        size = self.validate("size", 2, 10)
        print("(r)ed vs. (b)lue")
        print("blue goes from left to right, red goes from top to bottom.")
        print("which color will you be? Keep in mind that red will start (red=0, blue=1)")
        # Ask the user which color he would like to play with
        color = self.validate("color", -1, 2)
        # Create a board
        board = HexBoard(size)
        return board, color

    # Asks the user to give valid coordinates to make a move on
    def get_coordinates(self):
        print("x = ",end="")
        # Ask the user for the x-coordinate
        x = input()
        # While it is an incorrect input, ask the user again
        while x == '':
            print("invalid x-coordinate!")
            print("x = ",end="")
            x = input()
        print("y = ",end="")
        # Ask the user for the y-coordinate
        y = input()
        # While it is an incorrect input, ask the user again
        while y == '':
            print("invalid y-coordinate!")
            print("y = ",end="")
            y = input()
        # Return the coordinates
        return int(x), int(y)

    # Lets the user make a move. 
    def makeMove(self,board, color):
        valid = False
        # While the user gives invalid coordinates, ask the user again
        while(not valid):
            # Get the coordinates
            x, y = self.get_coordinates()
            # Check if the coordinates are on the board
            if(not (0 <= x and x < board.size and 0 <= y and y < board.size)):
                print("Invalid coordinates!")
            else:
                # Check if the coordinates are empty
                if(board.is_empty((x,y))):
                    # If it's empty, place your color
                    board.place((x,y), color)
                    return True
                else:
                    print("Place already taken!")

    # Validates if an input is between upper and lower boundaries
    def validate(self, val, lower, upper):
        print(val+" = ", end="")
        res = input()
        # While the input is not between the upper and lower boundaries, ask the user again
        while(res == '' or not(lower < int(res) and int(res) < upper)):
            print("Invalid input!")
            res = input()
        res = int(res)
        return res