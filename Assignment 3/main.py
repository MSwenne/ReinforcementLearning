#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 3: Function Approximation       #
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

from mountain_car import Model as Model_mc
from breakout import Model as Model_br
from utils import get_input
import numpy as np
import gym

if __name__ == "__main__":
    print("Which part of the assignment would you like to see?")
    ans = get_input("(M)ountain car or (B)reakout", ['M', 'B', 'm', 'b'])
    if ans == 'M' or ans == 'm':
        env = gym.make('MountainCar-v0')
        model = Model_mc(np.array(env.observation_space.shape), int(env.action_space.n))
    if ans == 'B' or ans == 'b':
        env = gym.make('Breakout-v0')
        model = Model_br(np.array(env.observation_space.shape), int(env.action_space.n))
    model.train(iterations=100)
    obs = env.reset()
    for t in range(model.max_steps):
        obs, _, done, _ = env.step(model.predict(obs))
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    env.close()