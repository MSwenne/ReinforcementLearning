#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 2: HEX                          #
#   Main function                                                   #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 6 March 2020                                         #
# All rights reserved                                               #
#                                                                   #
#####################################################################


from trueskill import Rating, quality_1vs1, rate_1vs1
from hex_skeleton import HexBoard
from AlphaBeta import AlphaBeta
from MCTS_Hex import MCTS
from PlayGame import Play
from utils import get_input
import numpy as np

if __name__ == "__main__":
    print("Which part of the assignment would you like to see?")
    part = get_input("(M)CTS Hex, (E)xperiment, (T)une", ['M', 'E', 'T', 'm', 'e', 't'])
    if part == 'M' or part == 'm':
        part2 = get_input("(H)uman vs. MCTS or (M)inimax vs. MCTS?", ['H', 'M', 'h', 'm'])
        bot_MCTS = MCTS(Cp=np.sqrt(2), itermax=5000)
        bot_AB = AlphaBeta(depth=3)

        if part2 == 'H' or part2 == 'h':
            game = Play(player1=None, player2=bot_MCTS)
        if part2 == 'M' or part2 == 'm':
            starter = get_input("Who starts? (A)lpha-Beta or (M)CTS?", ['A', 'M', 'a', 'm'])
            p1 = bot_AB if starter == 'A' or starter == 'a' else bot_MCTS
            p2 = bot_MCTS if starter == 'A' or starter == 'a' else bot_AB
            game = Play(player1=p1, player2=p2)
        game.play_game()

    if part == 'E' or part == 'e':
        bot_MCTS = MCTS(Cp=np.sqrt(2), itermax=5000)
        bot_AB = AlphaBeta(depth=3)
        bots[]
        # Initialise the number of bots (max 3), rounds and board size
        rounds = 10
        size = 3
        print("board size: ", size," : ",rounds," rounds.")
        # Initialise color, printing params, depths and heuristics
        color = [HexBoard.BLUE, HexBoard.RED]
        heuristics = [[False, True], [False, True], [True, True]]
        # Initialise ratings
        r1 = Rating()
        r2 = Rating()
        print(r1, r2)
        # For each round:
        for round in range(rounds):
            print("round:", round+1, end=" ")
            # Switch starting player each round
            turn = 0 if round % 2 == 0 else 1
            if turn:
                print("MCTS starts")
            else:
                print("ALpha-Beta starts")
            # Setup board, depth and heuristic
            board = HexBoard(size)
            # While the game is not over
            while(not board.is_game_over()):
                # Make a ove using alpha_beta search and a random or Dijkstra eval
                bot_MCTS.mo(board, color[turn], depth[turn], heuristic[turn])
                # Switch turns
                turn = int(not turn)
            # Print board and winner after game ends
            board.print()
            if board.check_win(color[1]):
                print("RED wins! ", )
            else:
                print("BLUE wins! ", )
            # Update ratings accordingly
            if board.check_win(color[0]):
                if i == 0:
                    r1, r2 = rate_1vs1(r1, r2)
                if i == 1:
                    r1, r3 = rate_1vs1(r1, r3)
                if i == 2:
                    r2, r3 = rate_1vs1(r2, r3)
            else:
                if i == 0:
                    r2, r1 = rate_1vs1(r2, r1)
                if i == 1:
                    r3, r1 = rate_1vs1(r3, r1)
                if i == 2:
                    r3, r2 = rate_1vs1(r3, r2)
            # Print new ratings and clean up board
            print(r1, r2, r3)
            del board

    if part == 'T' or part == 't':
        pass

