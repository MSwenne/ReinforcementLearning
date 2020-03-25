#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 2: HEX                          #
#   Main function                                                   #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 24 March 2020                                        #
# All rights reserved                                               #
#                                                                   #
#####################################################################


from trueskill import Rating, quality_1vs1, rate_1vs1
from hex_skeleton import HexBoard
from AlphaBeta import AlphaBeta
from MCTS_Hex_pseudo import MCTS
from PlayGame import Play
from utils import get_input
import numpy as np

max_time = 0.1

if __name__ == "__main__":
    print("Which part of the assignment would you like to see?")
    ans = get_input("(M)CTS Hex, (E)xperiment, (T)une", ['M', 'E', 'T', 'm', 'e', 't'])
    if ans == 'M' or ans == 'm':
        part1()
    if ans == 'E' or ans == 'e':
        part2()
    if ans == 'T' or ans == 't':
        pass

def part1():
    ans = get_input("(H)uman vs. MCTS or (A)lpha-Beta vs. MCTS?", ['H', 'A', 'h', 'a'])
    bot_MCTS = MCTS(Cp=np.sqrt(2), itermax=5000, max_time=max_time)
    bot_AB = AlphaBeta(depth=3, max_time=max_time)

    if ans == 'H' or ans == 'h':
        game = Play(player1=None, player2=bot_MCTS)
    if ans == 'A' or ans == 'a':
        starter = get_input("Who starts? (A)lpha-Beta or (M)CTS?", ['A', 'M', 'a', 'm'])
        p1 = bot_AB if starter == 'A' or starter == 'a' else bot_MCTS
        p2 = bot_MCTS if starter == 'A' or starter == 'a' else bot_AB
        game = Play(player1=p1, player2=p2)
    game.play_game()

def part2():
    bot_MCTS = MCTS(Cp=np.sqrt(2), itermax=5000, max_time=max_time)
    bot_AB = AlphaBeta(depth=3, max_time=max_time)
    bots = [bot_MCTS, bot_AB]
    color = [HexBoard.RED, HexBoard.BLUE]
    # Initialise the number of rounds and board size
    print("Hex game: how big is the board? (minimal 2x2 and maximal 10x10)")
    size = int(get_input("board size = ", [str(i) for i in range(2,11)], ""))
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
            bots[turn].makeMove(board, color[turn])
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
        print(r1, r2)
        del board

