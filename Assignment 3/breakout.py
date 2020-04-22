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

class Model:
    def __init__(self, env, input_size, output_size):
        self.low = env.observation_space.low
        self.high = env.observation_space.high
        self.max_steps = 200
        self.batch_size = 32
        self.memory = deque(maxlen=2000)
        self.discount_rate = 0.95
        self.optimizer = keras.optimizers.Adam(lr=1e-3)
        self.activation = 'relu'
        self.loss = keras.losses.mean_squared_error
        self.input_size = input_size
        self.output_size = output_size
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

    def build_model(self):
        model = keras.Sequential([
            keras.layers.Conv2D(filters=16, kernel_size=8, strides=4, activation=self.activation, input_shape=self.input_size),
            keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation=self.activation),
            keras.layers.Dense(256, activation=self.activation),
            keras.layers.Dense(self.output_size)
        ])
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def train(self, iterations=100):
        print("Training model:")
        for i in tqdm(range(iterations)):
            obs = make_gray(env.reset())
            for s in range(self.max_steps):
                epsilon = max(1 - i / iterations, 0.01)
                action = self.predict(obs, epsilon)
                next_obs, reward, done, _ = env.step(action)
                next_obs = make_gray(next_obs)
                self.memory.append((obs, next_obs, action, reward, done))
                obs = next_obs
                if s % 10 == 0:
                    self.training_step()
                if done:
                    break
            self.target_model.set_weights(self.model.get_weights())

    def predict(self, obs, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.output_size)
        else:
            Q_values = self.model.predict(obs)
            return np.argmax(Q_values[0])

    def training_step(self):
        if len(self.memory) < self.batch_size:
            return
        print("1")
        samples = np.array(random.sample(self.memory,self.batch_size))
        obs, next_obs, action, reward, done = np.hsplit(samples, 5)
        print("2")
        obs = [[item[0] for item in obs]]
        print("3")
        next_obs = [[item[0] for item in next_obs]] 
        print("4")
        action = action.reshape(self.batch_size,).astype(int)
        reward = reward.reshape(self.batch_size).astype(float)
        done = np.concatenate(done).astype(bool)
        not_done = (~done).astype(float)
        done = done.astype(float)
        print("5")
        target = self.model.predict(obs, verbose=1)
        print("6")
        Q_futures = self.target_model.predict(next_obs, verbose=2).max(axis = 1)
        print("7")
        target[(np.arange(self.batch_size), action)] = reward * done + (reward + Q_futures * self.discount_rate)*not_done
        print("8")

def make_gray(img):
    new_img = tf.image.rgb_to_grayscale(tf.convert_to_tensor(img), name=None)
    new_img = tf.image.per_image_standardization(new_img)
    return new_img/255

if __name__ == "__main__":
    env = gym.make('Breakout-v0')
    obs = make_gray(env.reset())
    model = Model(env, obs.shape, int(env.action_space.n))
    model.train(iterations=100)
    model.model.save_weights('./weights/weights')
    # model.model.load_weights('./weights/weights')
    model.target_model.save_weights('./weights/target_weights')
    # model.target_model.load_weights('./weights/target_weights')
    obs = make_gray(env.reset())
    for t in range(model.max_steps):
        action = model.predict(obs)
        print(action)
        obs, _, done, _ = env.step(action)
        obs = make_gray(obs)
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    env.close()
