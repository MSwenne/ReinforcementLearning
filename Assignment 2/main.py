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

if __name__ == "__main__":
    print("Which part of the assignment would you like to see?")
    print("(M)CTS Hex, (E)xperiment, (T)une")
    part = input()
    while part not in ['M', 'E', 'T']:
        part = input()


    
    bot = MCTS()
    game_player = Play(bot)
