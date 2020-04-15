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
from utils import update_scene
from utils import plot_animation
import numpy as np
import random
import gym

class Model:
    def __init__(self, env, input_size, output_size):
        self.max_steps = 200
        self.batch_size = 32
        self.discount_rate = 0.95
        self.optimizer = keras.optimizers.Adam(lr=1e-3)
        self.activation = 'elu'
        self.loss = keras.losses.mean_squared_error
        self.replay_memory = deque(maxlen=2000)
        self.input_size = input_size
        self.output_size = output_size
        self.model = self.build_model()

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
        max_pos = env.observation_space.high[0]
        best_score = -np.inf
        for i in tqdm(range(iterations)):
            obs = env.reset()
            game_score = 0
            for _ in range(self.max_steps):
                epsilon = max(1 - i / iterations, 0.01)
                action = self.predict(obs, epsilon)
                next_obs, reward, done, _ = env.step(action)
                self.replay_memory.append((obs, next_obs, action, reward, done))
                game_score += self.reward(next_obs, max_pos)
                if done:
                    break
                obs = next_obs
            if game_score > best_score:
                best_weights = self.model.get_weights()
                best_score = game_score
            if i > 20:
                self.training_step()
        self.model.set_weights(best_weights)

    def predict(self, obs, epsilon=0):
        if np.random.rand() < epsilon:
            return np.random.randint(self.output_size)
        else:
            Q_values = self.model.predict(obs[np.newaxis])
            return np.argmax(Q_values[0])

    def reward(self, obs, max_pos):
        pos, vel = obs
        return (1/(max_pos - pos)) * abs(vel)

    def training_step(self):
        obs, next_obs, actions, rewards, dones = self.sample_experiences()
        next_Q_values = self.model.predict(next_obs)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1 - dones) * self.discount_rate * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)
        mask = tf.one_hot(actions, self.output_size)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(obs)
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def sample_experiences(self):
        indices = np.random.randint(len(self.replay_memory), size=self.batch_size)
        batch = [self.replay_memory[index] for index in indices]
        return [np.array([experience[i] for experience in batch]) for i in range(len(batch[0]))]

if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    model = Model(env, env.observation_space.shape[0], int(env.action_space.n))
    model.train(iterations=100)
    obs = env.reset()
    frames =[]
    for t in range(model.max_steps):
        obs, _, done, _ = env.step(model.predict(obs))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
        img = env.render(mode="rgb_array")
        frames.append(img)
    env.close()
    plot_animation(frames)
