#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 1: HEX                          #
#   Part 1: Search                                                  #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 28 Februari 2020                                     #
# All rights reserved                                               #
#                                                                   #
#####################################################################

from hex_skeleton import HexBoard
import numpy as np
import random

# Allows the user to play a game of Hex versus a bot
# The bot uses alpha-beta search and a random evaluation
def play_game():
    #initialize the board size and player color
    board, player = init()
    turn = 0
    color = [HexBoard.RED, HexBoard.BLUE]
    # Run a game until it is over
    while(not board.is_game_over()):
        # If it is the turn of the player...
        if turn == player:
            print("make a move...")
            # ... let him make a move
            makeMove(board, color[turn])
        # If it is the turn of the bot...
        else:
            print("enemy's turn:")
            # Generate a move for the bot
            makeAlphaBetaMove(board, color[turn], 3)
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
def init():
    print("Hex game: how big is the board? (minimal 2x2 and maximal 10x10)")
    # Ask the user for a board size
    size = validate("size", 2, 10)
    print("(r)ed vs. (b)lue")
    print("blue goes from left to right, red goes from top to bottom.")
    print("which color will you be? Keep in mind that red will start (red=0, blue=1)")
    # Ask the user which color he would like to play with
    color = validate("color", -1, 2)
    # Create a board
    board = HexBoard(size)
    return board, color

# Asks the user to give valid coordinates to make a move on
def get_coordinates():
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
def makeMove(board, color):
    valid = False
    # While the user gives invalid coordinates, ask the user again
    while(not valid):
        # Get the coordinates
        x, y = get_coordinates()
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

# Returns all empty coordinates
def getMoveList(board, color):
    # Initialise an empty list of empty coordinates
    moves = []
    for x in range(board.size):
        for y in range(board.size):
            # If the coordinates are empty...
            if(board.is_empty((x,y))):
                # ... append the coordinates to the list
                moves.append((x,y))
    # Return the list of empty coordinates
    return moves

# Validates if an input is between upper and lower boundaries
def validate(val, lower, upper):
    print(val+" = ", end="")
    res = input()
    # While the input is not between the upper and lower boundaries, ask the user again
    while(res == '' or not(lower < int(res) and int(res) < upper)):
        print("Invalid input!")
        res = input()
    res = int(res)
    return res

# Makes a move that uses alpha-beta search and a random eval
def makeAlphaBetaMove(board, color, depth):
    # Initialise the enemy, best_value and best_move
    enemy = board.get_opposite_color(color)
    best_value = np.inf
    best_move = 0
    # For every possible move on the board...
    for move in getMoveList(board, color):
        # ...play this move...
        board.place(move, color)
        # ...do alpha-beta search...
        value = alpha_beta(board, depth, -np.inf, np.inf, enemy, True)
        # ...and undo the move again
        board.unplace(move)
        # If the value of the alpha-beta search is better that your current best_value...
        if(value < best_value):
            # ...update the best_value and the best_move
            best_move = move
            best_value = value
    # If there is no best_move (it is still 0), just take the last viewed move
    if best_move == 0:
        best_move = move
    # Place the best_move
    board.place(best_move, color)

# Does n-depth alpha_beta search on a board with a random eval
def alpha_beta(board, depth, alpha, beta, color, maximize):
    enemy = board.get_opposite_color(color)
    # If depth is 0 or the game ended, eval the board
    if board.check_win(color) == 0:
        return 1
    if board.check_win(enemy) == 0:
        return 0
    if depth == 0:
        return random.random()
    # If this is the maximizing player...
    if maximize:
        # ...initialise the value...
        value = -np.inf
        # ...and for each possible move...
        for move in getMoveList(board, color):
            # ...play the move...
            board.place(move, color)
            # ...recursively call alpha_beta search on the enemy...
            value = max(value, alpha_beta(board, depth-1, alpha, beta, enemy, False))
            # ...undo the move...
            board.unplace(move)
            # ...and update the alpha value
            alpha = max(alpha, value)
            # If alpha is greater or equal to beta, break
            if (alpha >= beta):
                break
    # If this is the minimizing player...
    else:
        # ...initialise the value...
        value = np.inf
        # ...and for each possible move...
        for move in getMoveList(board, color):
            # ...play the move...
            board.place(move, color)
            # ...recursively call alpha_beta search on the enemy...
            value = min(value, alpha_beta(board, depth-1, alpha, beta, enemy, True))
            # ...undo the move...
            board.unplace(move)
            # ...and update the alpha value
            beta = min(beta, value)
            # If alpha is greater or equal to beta, break
            if (alpha >= beta):
                break
    # Return the eval value
    return value

# Calls play a game
if __name__ == "__main__":
    play_game()