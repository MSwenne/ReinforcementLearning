#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 3: Function Approximation       #
#   Mountain Car function                                           #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 22 April 2020                                        #
# All rights reserved                                               #
#                                                                   #
#####################################################################

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from collections import deque
import numpy as np
import random
import gym

WIN_DATA = []
SCORE_DATA = []

class Model:
    def __init__(self, env):
        self.env = env
        self.discount_rate = 0.99
        self.max_steps = 201
        self.batch_size = 32
        self.memory = deque(maxlen=20000)
        self.optimizer = keras.optimizers.Adam(lr=0.001)
        self.activation = 'relu'
        self.loss = 'mse'
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = int(self.env.action_space.n)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, activation=self.activation, input_dim=self.input_size),
            keras.layers.Dense(48, activation=self.activation),
            keras.layers.Dense(self.output_size, activation='linear')
        ])
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def train(self, iterations=100):
        print('Training model:')
        for i in range(iterations):
            rewardSum = 0
            max_position=-99
            obs = self.env.reset().reshape(1, 2)
            for s in range(self.max_steps):
                if i % 50 == 0:
                    self.env.render()
                epsilon = max(1 - i * 0.05, 0.01)
                action = self.predict(obs, epsilon)
                # print(action, end='')
                next_obs, reward, done, _ = self.env.step(action)
                next_obs = next_obs.reshape(1, 2)
                if next_obs[0][0] > max_position:
                    max_position = next_obs[0][0]
                if next_obs[0][0] >= 0.5:
                    reward += 10
                self.memory.append([obs, next_obs, action, reward, done])
                self.training_step()
                rewardSum += reward
                obs = next_obs
                # if (s+1) % 50 == 0:
                #     print()
                if done:
                    # print()
                    break
            if s >= 199:
                print('Failed to finish task in iteration {}/{}'.format(i,iterations))
                WIN_DATA.append(False)
            else:
                print()
                print('Success in iteration {}/{}, used {} steps!'.format(i, iterations, s))
                WIN_DATA.append(True)
            SCORE_DATA.append(rewardSum)
            self.target_model.set_weights(self.model.get_weights())
            # print('\tnow epsilon is {}, the reward is {} maxPosition is {}'.format(round(epsilon,2), rewardSum, max_position))

    def predict(self, obs, epsilon=0):
        if np.random.rand(1) < epsilon:
            action = np.random.randint(self.output_size)
        else:
            Q_values = self.model.predict(obs)
            action = np.argmax(Q_values[0])
        return action

    def training_step(self):
        if len(self.memory) < self.batch_size:
            return
        samples = random.sample(self.memory,self.batch_size)
        npsamples = np.array(samples)
        states_temp, newstates_temp, actions_temp, rewards_temp, dones_temp = np.hsplit(npsamples, 5)
        states = np.concatenate((np.squeeze(states_temp[:])), axis = 0)
        rewards = rewards_temp.reshape(self.batch_size,).astype(float)
        targets = self.model.predict(states)
        newstates = np.concatenate(np.concatenate(newstates_temp))
        dones = np.concatenate(dones_temp).astype(bool)
        notdones = ~dones
        notdones = notdones.astype(float)
        dones = dones.astype(float)
        Q_futures = self.target_model.predict(newstates).max(axis = 1)
        targets[(np.arange(self.batch_size), actions_temp.reshape(self.batch_size,).astype(int))] = rewards * dones + (rewards + Q_futures * self.discount_rate)*notdones
        self.model.fit(states, targets, epochs=1, verbose=0)

    def play_one_game(self):
        # print()
        obs = self.env.reset().reshape(1,2)
        rewardSum = 0
        for t in range(self.max_steps):
            action = self.predict(obs)
            # print(action,end=', ')
            # if (t+1) % 50 == 0:
            #     print()
            obs, reward, done, _ = self.env.step(action)
            rewardSum += reward
            obs = obs.reshape(1,2)
            self.env.render()
            if done:
                print('\nEpisode finished after {} timesteps'.format(t+1))
                break
        WIN_DATA.append(t < 199)
        SCORE_DATA.append(rewardSum)

    def save_weights(self, file):
        self.model.save_weights(file)

    def load_weights(self, file):
        self.model.load_weights(file)

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    model = Model(env)
    model.train(iterations=500)
    # model.load_weights('./weights')
    for _ in range(50):
        model.play_one_game()
    model.save_weights('./weights')
    env.close()

    f = open('./data.txt','w+')
    string1 = ''
    string1 += str(WIN_DATA) + ' , '
    string1 += '\n'
    f.write(string1)
    string2 = ''
    string2 += str(SCORE_DATA) + ' , '
    string2 += '\n'
    f.write(string2)
    f.close()

    lose = [i for i, x in enumerate(WIN_DATA) if not x]
    win = [i for i, x in enumerate(WIN_DATA) if x]
    score_lose = [i for i, x in enumerate(SCORE_DATA) if x <= -199]
    score_win = [i for i, x in enumerate(SCORE_DATA) if x > -199]
    fig,a =  plt.subplots(2,1)
    a[0].scatter(score_lose,[SCORE_DATA[i] for i in score_lose] , c='r')
    a[0].scatter(score_win,[SCORE_DATA[i] for i in score_win] , c='g')
    a[0].plot([np.mean(SCORE_DATA)]*len(SCORE_DATA))
    a[0].set_title('Scores')
    a[1].scatter(lose,[WIN_DATA[i] for i in lose] , c='r')
    a[1].scatter(win,[WIN_DATA[i] for i in win] , c='g')
    a[1].set_title('Wins')
    plt.sca(a[0])
    plt.xticks(range(0,len(WIN_DATA),10),range(0,len(WIN_DATA),10))
    plt.yticks([0.0, -200.0], ["0","-200"])
    plt.sca(a[1])
    plt.xticks(range(0,len(WIN_DATA),10),range(0,len(WIN_DATA),10))
    plt.yticks([1.0, 0.0], ["True","False"])
    plt.ylabel('Score')
    # plt.legend(['score', 'win'])
    fig.tight_layout(pad=3.0)
    plt.savefig('stats.png')
    plt.show()
