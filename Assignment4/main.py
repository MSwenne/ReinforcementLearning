#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 4: Self-Play                    #
#   Main function                                                   #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 9 May 2020                                           #
# All rights reserved                                               #
#                                                                   #
#####################################################################


from trueskill import Rating, quality_1vs1, rate_1vs1
from hex_skeleton import HexBoard
from ParallelAlphaBeta import AlphaBeta
from operator import itemgetter 
from MCTS_Hex import MCTS
from utils import get_input
import numpy as np
from tune import tune

MAX_TIME = 0.2
CP = np.sqrt(4)
ITERMAX = 10000

def Tournament():
    bot_MCTS = MCTS(Cp=CP, itermax=ITERMAX, max_time=MAX_TIME)
    bot_AB = AlphaBeta(max_time=MAX_TIME)
    bots = [bot_MCTS, bot_AB]
    color = [HexBoard.RED, HexBoard.BLUE]
    # Initialise the number of rounds and board size
    print("Hex game: 7x7)")
    size = 7
    rounds = 10
    print(rounds,"rounds")
    print("MCTS vs. Alpha-Beta")
    # Initialise ratings
    r1 = Rating()
    r2 = Rating()
    print(r1, r2)
    # For each round:
    for round in range(rounds):
        print("round:", round+1, end="")
        # Switch starting player each round
        turn = 0 if round % 2 == 0 else 1
        if not turn:
            print(" - MCTS starts       - ", end="")
        else:
            print(" - Alpha-Beta starts - ", end="")
        # Setup board, depth and heuristic
        board = HexBoard(size)
        # While the game is not over
        while(not board.is_game_over()):
            # Make a move using corresponding bot
            board = bots[turn].makeMove(board, color[turn])
            # Switch turns
            turn = int(not turn)
        # Print board and winner after game ends
        if board.check_win(HexBoard.RED):
            print("MCTS wins!")
        else:
            print("Alpha-Beta wins!")
        # Update ratings accordingly
        if board.check_win(HexBoard.RED):
            r1, r2 = rate_1vs1(r1, r2)
        else:
            r2, r1 = rate_1vs1(r2, r1)
        # Print new ratings and clean up board
        del board
    print(r1, r2)

if __name__ == "__main__":
    print("Which part of the assignment would you like to see?")
    ans = get_input("(M)CTS Hex, (E)xperiment, (T)une", ['M', 'E', 'T', 'm', 'e', 't'])
    if ans == 'T' or ans == 't':
        Tournament()