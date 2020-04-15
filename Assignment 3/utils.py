#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 3: Function Approximation       #
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

import matplotlib.animation as animation
import matplotlib.pyplot as plt

def get_input(message, valid, ending="\n"):
    print(message, end=ending)
    result = input()
    while result not in valid:
        print("Invalid value!")
        print(message, end=ending)
        result = input()
    return result

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim