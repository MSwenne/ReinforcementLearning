#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 3: Function Approximation       #
#   Mountain Car function                                           #
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
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 2 , 'CPU': 1} ) 
sess = tf.compat.v1.Session(config=config) 
tf.compat.v1.keras.backend.set_session(sess)

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
            keras.layers.Dense(32, activation=self.activation, input_dim=self.input_size),
            keras.layers.Dense(32, activation=self.activation),
            keras.layers.Dense(self.output_size)
        ])
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def train(self, iterations=100):
        print("Training model:")
        for i in tqdm(range(iterations)):
            obs = env.reset()
            for s in range(self.max_steps):
                epsilon = max(1 - i / iterations, 0.01)
                action = self.predict(obs, epsilon)
                next_obs, reward, done, _ = env.step(action)
                self.memory.append((obs, next_obs, action, reward, done))
                # if s % 10 == 0:
                self.training_step()
                obs = next_obs
                if done:
                    break
            # if i % 10 == 0:
            self.target_model.set_weights(self.model.get_weights())

    def predict(self, obs, epsilon=0):
        if np.random.rand() < epsilon:
            action = np.random.randint(self.output_size)
            # print('random ',end='')
        else:
            Q_values = self.model.predict([[obs]])
            action = np.argmax(Q_values[0])
            # print('predict ',end='')
        # print(action)
        return action

    def training_step(self):
        if len(self.memory) < self.batch_size:
            return
        samples = np.array(random.sample(self.memory,self.batch_size))
        obs, next_obs, action, reward, done = np.hsplit(samples, 5)
        obs = [[item[0] for item in obs]]
        next_obs = [[item[0] for item in next_obs]]
        action = action.reshape(self.batch_size,).astype(int)
        reward = reward.reshape(self.batch_size).astype(float)
        done = np.concatenate(done).astype(bool)
        not_done = (~done).astype(float)
        done = done.astype(float)
        target = self.model.predict(obs)
        Q_futures = self.target_model.predict(next_obs).max(axis = 1)
        target[(np.arange(self.batch_size), action)] = reward * done + (reward + Q_futures * self.discount_rate)*not_done
        self.model.fit(obs, target, epochs=1, verbose=0)

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    env.seed(42)
    np.random.seed(42)
    tf.random.set_seed(42)
    obs = env.reset()
    model = Model(env, obs.shape[0], int(env.action_space.n))
    model.model.load_weights('./weights/weights')
    model.target_model.load_weights('./weights/target_weights')
    model.train(iterations=100)
    model.model.save_weights('./weights/weights')
    model.target_model.save_weights('./weights/target_weights')
    obs = env.reset()
    for t in range(model.max_steps):
        action = model.predict(obs)
        print(action)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    env.close()
