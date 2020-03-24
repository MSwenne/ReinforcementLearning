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

def get_input(message, valid, ending="\n"):
    print(message, end=ending)
    result = input()
    while result not in valid:
        print("Invalid value!")
        print(message, end=ending)
        result = input()
    return result
