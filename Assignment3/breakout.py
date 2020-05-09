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

import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from collections import deque
import numpy as np
import random
import gym
import time
from datetime import datetime

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
        self.input_size = self.make_gray(env.reset()).shape
        self.output_size = int(self.env.action_space.n)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Conv2D(filters=16, kernel_size=8, strides=4, activation=self.activation, input_shape=self.input_size),
            keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation=self.activation),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation=self.activation),
            keras.layers.Dense(self.output_size, activation='linear')
        ])
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def train(self, iterations=100):
        print('Training model:')
        for i in range(iterations):
            rewardSum = 0
            obs = self.make_gray(self.env.reset())
            for s in range(self.max_steps):
                if i % 50 == 0:
                    self.env.render()
                epsilon = max(1 - i * 0.05, 0.01)
                action = self.predict(obs, epsilon)
                print(action, end='')
                next_obs, reward, done, _ = self.env.step(action)
                next_obs = self.make_gray(next_obs)
                self.memory.append([obs, next_obs, action, reward, done])
                self.training_step()
                rewardSum += reward
                obs = next_obs
                if (s+1) % 50 == 0:
                    print()
                if done:
                    print()
                    break
            self.target_model.set_weights(self.model.get_weights())
            print('\tnow epsilon is {}, the reward is {}'.format(round(epsilon,2), rewardSum))

    def predict(self, obs, epsilon=0):
        if np.random.rand() < epsilon:
            action = np.random.randint(self.output_size)
        else:
            Q_values = self.model.predict(obs)
            action = np.argmax(Q_values[0])
        return action

    def training_step(self):
        if len(self.memory) < self.batch_size:
            return
        start = datetime.now()
        samples = random.sample(self.memory,self.batch_size)
        # npsamples = np.array(samples)
        states_temp, newstates_temp, actions_temp, rewards_temp, dones_temp = zip(*samples)
        states = np.array(states_temp).reshape((self.batch_size, ) + self.input_size)
        actions = list(actions_temp)
        rewards = [list(rewards_temp)]
        targets = self.model.predict(states)
        newstates = [list(newstates_temp)]
        dones = np.array(dones_temp)
        notdones = ~dones
        notdones = notdones.astype(float)
        dones = dones.astype(float)
        Q_futures = self.target_model.predict(newstates).max(axis = 1)
        targets[(np.arange(self.batch_size), actions)] = rewards * dones + (rewards + Q_futures * self.discount_rate)*notdones
        self.model.fit(states, targets, epochs=1, verbose=0)
        print('time is {}'.format(datetime.now() - start))

    def make_gray(self, img):
        new_img = tf.image.rgb_to_grayscale(tf.convert_to_tensor(img), name=None)
        new_img = tf.image.per_image_standardization(new_img)
        return new_img/255

    def play_one_game(self):
        print()
        obs = self.make_gray(self.env.reset())
        for t in range(self.max_steps):
            action = self.predict(obs)
            print(action,end=', ')
            if (t+1) % 50 == 0:
                print()
            obs, _, done, _ = self.env.step(action)
            obs = self.make_gray(obs)
            self.env.render()
            if done:
                print('\nEpisode finished after {} timesteps'.format(t+1))
                break

    def save_weights(self, file):
        self.model.save_weights(file)

    def load_weights(self, file):
        self.model.load_weights(file)

if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    model = Model(env)
    model.train(iterations=200)
    # model.load_weights('./weights')
    model.play_one_game()
    model.save_weights('./weights')
    env.close()
