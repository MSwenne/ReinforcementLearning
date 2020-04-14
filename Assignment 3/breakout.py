#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 3: Function Approximation       #
#   Breakout function                                               #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 24 March 2020                                        #
# All rights reserved                                               #
#                                                                   #
#####################################################################

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense
from tqdm import tqdm
import numpy as np
import random
import gym

env = gym.make('Breakout-v0')
env.reset()
train_games = 100
max_steps = 20000

class Model:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.input_size, activation='relu'))
        model.add(Dense(52, activation='relu'))
        model.add(Dense(self.output_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    def train(self, iterations=1000):
        print("Training model:")
        for _ in tqdm(range(iterations)):
            for _ in range(max_steps):
                observation, _, done, _ = env.step(env.action_space.sample())
                if done:
                    break
            env.reset()

    def predict(self, observation):
        pass

if __name__ == "__main__":
    model = Model(np.array(env.observation_space.shape), int(env.action_space.n))
    model.train(iterations=1000)
    for t in range(max_steps):
        env.render()
        [observation, score, done, _] = env.step(model.predict(observation))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    env.close()