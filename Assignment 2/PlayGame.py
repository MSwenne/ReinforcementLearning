#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 2: HEX                          #
#   Play class                                                      #
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
from utils import get_input
import numpy as np
import random
import heapq


class Play():
    def __init__(self, player2, player1=None):
        self.p1 = player1 # if None, human play
        self.p2 = player2
        self.board = None
        self.size = 0

    # Allows the user to play a game of Hex versus a bot
    # The bot uses alpha-beta search and a random evaluation
    def play_game(self):
        #initialize the board size and player color
        player = self.init_params()
        turn = 0
        color = [HexBoard.RED, HexBoard.BLUE]

        # Run a game until it is over
        while(not self.board.is_game_over()):
            # If it is the turn of the player...
            if turn == player:
                if self.p1 == None:
                    print("make a move...")
                    # ... let him make a move
                    self.makeMove(color[turn])
                else:
                    self.board = self.p1.makeMove(color[turn])
            # If it is the turn of the bot...
            else:
                print("enemy's turn:")
                # Generate a move for the bot
                self.board = self.p2.makeMove(self.board, color[turn])
            # Print the board after every move
            self.board.print()
            # Switch turns
            turn = int(not turn)
        # If the game is over, show who has won
        if self.board.check_win(color[0]):
            print("RED wins!")
        else:
            print("BLUE wins!")
        del self.board

    # Function that asks questions to the user about the game parameters
    def init_params(self):
        print("Hex game: how big is the board? (minimal 2x2 and maximal 10x10)")
        # Ask the user for a board size
        self.size = int(get_input("size = ", [str(i) for i in range(2,11)], ""))
        if self.p1 == None:
            print("(r)ed vs. (b)lue")
            print("blue goes from left to right, red goes from top to bottom.")
            print("which color will you be? Keep in mind that red will start (red=0, blue=1)")
            # Ask the user which color he would like to play with
            player = int(get_input("color = ", [str(i) for i in range(0,2)], ""))
        else:
            # bot1 plays red
            player = 0
        # Create a board
        self.board = HexBoard(self.size)
        return player

    # Asks the user to give valid coordinates to make a move on
    def get_coordinates(self):
        # Ask the user for the x-coordinate
        x = get_input("x = ", [str(i) for i in range(self.size)],"")
        # Ask the user for the y-coordinate
        y = get_input("y = ", [str(i) for i in range(self.size)],"")
        # Return the coordinates
        return int(x), int(y)

    # Lets the user make a move. 
    def makeMove(self, color):
        # While the user gives invalid coordinates, ask the user again
        while True:
            # Get the coordinates
            x, y = self.get_coordinates()
            # Check if the coordinates are on the board
            if(not (0 <= x and x < self.size and 0 <= y and y < self.size)):
                print("Invalid coordinates!")
            else:
                # Check if the coordinates are empty
                if(self.board.is_empty((x,y))):
                    # If it's empty, place your color
                    self.board.place((x,y), color)
                    return True
                else:
                    print("Place already taken!")