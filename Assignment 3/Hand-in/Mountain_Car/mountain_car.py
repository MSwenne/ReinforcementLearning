#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 3: Function Approximation       #
#   Mountain Car function                                           #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 24 April 2020                                        #
# All rights reserved                                               #
#                                                                   #
#####################################################################

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from collections import deque
import numpy as np
import random
import gym
import os

class Model:
    # Initialize parameters
    def __init__(self, env, batch_size=32, dr=0.99, memory_size=20000, lr=0.001, activation='relu', loss='mse'):
        self.win_data = []
        self.score_data = []
        self.env = env
        self.batch_size = batch_size
        self.discount_rate = dr
        self.max_steps = 201 # maximum steps of mountain_car is 200
        self.memory = deque(maxlen=memory_size)
        self.optimizer = keras.optimizers.Adam(lr=lr)
        self.activation = activation
        self.loss = loss
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = int(self.env.action_space.n)
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.target_model.set_weights(self.model.get_weights())

    # Build the neural network model
    def build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(24, activation=self.activation, input_dim=self.input_size),
            keras.layers.Dense(48, activation=self.activation),
            keras.layers.Dense(self.output_size, activation='linear')
        ])
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    # Train the model
    def train(self, episodes=100):
        print('Training model:')
        # Train for a number of episodes
        for episode in range(episodes):
            # Reset the env and the rewardSum
            rewardSum = 0
            state = self.env.reset().reshape(1, 2)
            # Do a playout
            for step in range(self.max_steps):
                # Show game every 50 episodes
                if episode % 50 == 0:
                    self.env.render()
                # Calculate the epsilon for the given episode
                epsilon = max(1 - episode * 0.05, 0.01)
                # Predict an action using the epsilon randomness
                action = self.predict(state, epsilon)
                # Perform action
                next_state, reward, done, _ = self.env.step(action)
                next_state = next_state.reshape(1, 2)
                # If it wins, make the reward higher
                if next_state[0][0] >= 0.5:
                    reward += 10
                # Store env results in memory
                self.memory.append([state, next_state, action, reward, done])
                # Perform a training step
                self.training_step()
                # Add reward to rewardSum
                rewardSum += reward
                # Update current state
                state = next_state
                # If done, stop playout
                if done:
                    break
            # Print if we won or not
            if step >= 199:
                print('Failure in iteration {}/{}'.format(episode+1,episodes))
            else:
                print('Success in iteration {}/{}, used {} steps!'.format(episode+1, episodes, step))
            # Store data for plotting
            self.win_data.append(step < 199)
            self.score_data.append(rewardSum)
            # Update weights of the model
            self.target_model.set_weights(self.model.get_weights())

    # Predict an action using epsilon randomness
    def predict(self, obs, epsilon=0):
        # If epsilon > 0 there is a chance to pick a random value
        # This is used for training, but not for testing
        if np.random.rand(1) < epsilon:
            action = np.random.randint(self.output_size)
        else:
            # Predict an action using the model
            Q_values = self.model.predict(obs)
            action = np.argmax(Q_values[0])
        return action

    # Do a training step using a batch from the memory
    def training_step(self):
        # If there is not enough memory to fill the batch, return
        if len(self.memory) < self.batch_size:
            return
        # Get a number of samples from the memory
        samples = np.array(random.sample(self.memory,self.batch_size))
        # Split the samples into the different parameters
        states, newstates, actions, rewards, dones = np.hsplit(samples, 5)
        # Parse all parameters to the correct shape
        states = np.concatenate((np.squeeze(states[:])), axis = 0)
        newstates = np.concatenate(np.concatenate(newstates))
        actions = actions.reshape(self.batch_size,).astype(int)
        rewards = rewards.reshape(self.batch_size,).astype(float)
        dones = np.concatenate(dones).astype(bool)
        not_dones = (~dones).astype(float)
        dones = dones.astype(float)
        # Predict the targets and Q_futures
        targets = self.model.predict(states)
        Q_futures = self.target_model.predict(newstates).max(axis = 1)
        # Update the target values using Q-learning
        targets[(np.arange(self.batch_size), actions)] = rewards * dones + (rewards + Q_futures * self.discount_rate)*not_dones
        # Fit the model to the new target values
        self.model.fit(states, targets, epochs=1, verbose=0)

    # Play one game normally (without epsilon randomness)
    def play_one_game(self):
        print('\nChosen actions:')
        state = self.env.reset().reshape(1,2)
        rewardSum = 0
        # Do a playout
        for step in range(self.max_steps):
            # Predict an action without epsilon randomness
            action = self.predict(state)
            # Print 50 actions per line
            print(action,end=', ')
            if (step+1) % 50 == 0:
                print()
            # Perform action
            state, reward, done, _ = self.env.step(action)
            # Add reward to rewardSum
            rewardSum += reward
            state = state.reshape(1,2)
            # Render current state
            self.env.render()
            # If done, stop playout
            if done:
                break
        # Print if we won or not
        if step >= 199:
            print('Failed to reach flag in 200 steps.')
        else:
            print('Success, used {} steps!'.format(step))
        # Store data for plotting
        self.win_data.append(step < 199)
        self.score_data.append(rewardSum)

    # Store weights
    def save_weights(self, file):
        self.model.save_weights(file)

    # Load previously saved weights
    def load_weights(self, file):
        self.model.load_weights(file)

    # Write wins and scores to .txt file
    def write_data(self):
        # If no previous data folder exists, create folder to put it in
        if not os.path.exists('./datafiles'):
            os.makedirs('./datafiles')
        # If previous data files exist, add a number behind the filename
        counter = 0
        filename = './datafiles/data{}.txt'
        while os.path.isfile(filename.format(counter)):
            counter += 1
        filename = filename.format(counter)
        # Write the data to the file
        f = open(filename,'w+')
        f.write(str(int(self.win_data)) + '\n')
        f.write(str(self.score_data) + '\n')
        f.close()
        print('Data written to {}.'.format(filename))
        # Return filename
        return filename

    # Read the data from a saved data file
    def read_data(self, file):
        # Open file and read data
        f = open('./datafiles/'+file,'r')
        data =  f.readlines()
        win_data = data[0].split()
        score_data = data[1].split()
        # Parse win data
        for i, data in enumerate(win_data):
            if i == 0:
                win_data[i] = int(data[1:-1] == 'True')
            else:
                win_data[i] = int(data[:-1] == 'True')
        # Parse score data
        for i, data in enumerate(score_data):
            if i == 0:
                score_data[i] = float(data[1:-1])
            else:
                score_data[i] = float(data[:-1])
        # Return data
        return win_data, score_data

    # Plot a figure with the data
    def plot_fig(self, win_data=None, score_data=None):
        # If no data is given, use the data from the model
        if not win_data:
            win_data = self.win_data
        if not win_data:
            win_data = self.win_data
        # Calculate the rolling window avg of the last 50 games
        avg_win = [np.mean(win_data[i:i+50]) for i in range(len(win_data)-50)]
        avg_score = [np.mean(score_data[i:i+50]) for i in range(len(score_data)-50)]
        # Separate loss and wins
        lose = [i for i, x in enumerate(win_data) if not x]
        win = [i for i, x in enumerate(win_data) if x]
        score_lose = [i for i, x in enumerate(score_data) if x <= -199]
        score_win = [i for i, x in enumerate(score_data) if x > -199]
        # create 2 plots in a figure and set title
        fig,a =  plt.subplots(2,1,figsize=(10, 10), dpi=120, facecolor='w', edgecolor='k')
        fig.suptitle('Mountain Car DQN training', size=14)
        # Fill plot 1 with data and mean and set title
        a[0].scatter(score_lose,[score_data[i] for i in score_lose] , c='r', s=4)
        a[0].scatter(score_win,[score_data[i] for i in score_win] , c='g', s=4)
        a[0].plot(range(50,len(score_data)), avg_score, c='b', label='mean of last 50 games', markersize=4)
        a[0].set_title('Mountain Car DQN scores', size=12)
        # Fill plot 2 with data and mean and set title
        a[1].scatter(lose,[win_data[i] for i in lose] , c='r', s=4)
        a[1].scatter(win,[win_data[i] for i in win] , c='g', s=4)
        a[1].plot(range(50,len(win_data)), avg_win, c='b', label='mean', markersize=4)
        a[1].set_title('Mountain Car DQN wins', size=12)
        # Edit plot 1 looks
        plt.sca(a[0])
        plt.xticks(range(0,len(win_data)+1,int(len(win_data)/11)),range(0,len(win_data)+1,int(len(win_data)/11)))
        plt.grid(True)
        plt.yticks([0, -50, -100, -150, -200], ['0','-50','-100','-150','-200'])
        plt.legend(framealpha=1, frameon=True, loc='upper right',prop={'size': 12})
        plt.xlabel('Episode')
        plt.ylabel('Score')
        # Edit plot 2 looks
        plt.sca(a[1])
        plt.grid(True)
        plt.xticks(range(0,len(win_data)+1,int(len(win_data)/11)),range(0,len(win_data)+1,int(len(win_data)/11)))
        plt.yticks([1.0, 0.75, 0.5, 0.25, 0.0], ['100%','75%','50%','25%','0%'])
        plt.xlabel('Episode')
        plt.ylabel('Win')
        # Make plots tight fit
        fig.tight_layout(pad=4)
        # Save and show figure
        plt.savefig('stats.png')
        plt.show()

if __name__ == '__main__':
    model = Model(gym.make('MountainCar-v0'))
    model.train(episodes=1000)
    for _ in range(100):
        model.play_one_game()
    model.env.close()
    filename = model.write_data()
    model.plot_fig(model.read_data(filename))