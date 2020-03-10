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



from hex_skeleton import HexBoard
from AlphaBeta import AlphaBeta
from MCTS_Hex import MCTS
from PlayGame import Play
import numpy as np

if __name__ == "__main__":
    print("Which part of the assignment would you like to see?")
    print("(M)CTS Hex, (E)xperiment, (T)une")
    part = input()
    while part not in ['M', 'E', 'T', 'm', 'e', 't']:
        print("Invalid value!")
        part = input()

    if part == 'M' or part == 'm':
        print("(H)uman vs. MCTS or (M)inimax vs. MCTS?")
        part2 = input()
        while part2 not in ['H', 'M', 'h', 'm']:
            print("Invalid value!")
            part2 = input()

        bot_MCTS = MCTS(Cp=np.sqrt(2), itermax=5000)
        bot_AB = AlphaBeta(depth=3)

        if part2 == 'H' or part2 == 'h':
            game = Play(player1=None, player2=bot_MCTS)
        if part2 == 'M' or part2 == 'm':
            print("Who starts? (A)lpha-Beta or (M)CTS?")
            starter = input()
            while part2 not in ['A', 'M', 'a', 'm']:
                print("Invalid value!")
                starter = input()
            p1 = bot_AB if starter == 'A' or starter == 'a' else bot_MCTS
            p2 = bot_MCTS if starter == 'A' or starter == 'a' else bot_AB
            game = Play(player1=p1, player2=p2)

    if part == 'E' or part == 'e':
        pass

    if part == 'T' or part == 't':
        pass

    game.play_game()