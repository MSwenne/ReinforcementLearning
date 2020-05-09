#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 3: Function Approximation       #
#   Breakout                                                        #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 27 April 2020                                        #
# All rights reserved                                               #
#                                                                   #
#####################################################################


# Modules
import gym
import functools
import sys
import random
from collections import deque
from datetime import datetime

# Tensorflow
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
import tensorflow.keras.backend as K

# Numpy
import numpy as np
from statistics import mean 

# Local imports
from utils import make_gray, lambda_out_shape, intial_state_preprocess, state_preprocess



# Initial multi GPU
tf.compat.v1.disable_eager_execution()
tf.distribute.MirroredStrategy()

# Down sampling of each frame.
FRAME_WIDTH = 84                # Original size : 160
FRAME_HEIGHT = 84               # Original size : 210
TRAIN_INTERVAL = 4              # How often to perform a gradient step
NUM_REPLAY_MEMORY = 1000        # Size of the replay buffer

START_FROM = 0
Conv2D = keras.layers.Conv2D
Dense = keras.layers.Dense
Input = keras.layers.Input
Flatten = keras.layers.Flatten
LeakyReLU = keras.layers.LeakyReLU
Lambda = keras.layers.Lambda
Multiply = keras.layers.Multiply

class Agent(object):
    def __init__(self):
        self.batch_size = 32
        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=logdir,
            histogram_freq=0,
            batch_size=self.batch_size,
            write_graph=True,
            write_grads=True
        ) 
        self.total_loss = 0.0
        self.training_threshold = 20 + START_FROM                       # Start training from this episode.
        self.episods = 500                                              # Total number of episodes
        self.gamma = 0.99                                               # Discount factor in Bellman equation
        self.optimizer = keras.optimizers.Adam(lr=0.0001)               # Adam optimizer for DQN
        self.loss_fn = keras.losses.mean_squared_error                  # Mean squared error to calculat loss of networks
        self.env = gym.make('Breakout-v0')
        self.env.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        self.rewards = []
        self.best_score = -np.inf
        self.input_shape = self.env.observation_space.shape
        self.n_outputs =  self.env.action_space.n
        self.stack_size = 4                                             # Number of images to stack on top of eachother. 

        self.replay_memory = deque(maxlen=NUM_REPLAY_MEMORY)            # Size of replay buffer
        self.epsilon = 0                                                # Initial value for eplison

        self.q_network = self.build_network()                           # Q Network to estimate Q-values
        self.target_network = self.build_network()                      # Target network to estimate best action
        
        self.dummy_input = np.zeros((1,self.n_outputs))                 # Inspired from the blogpost cited in report
        self.dummy_batch = np.zeros((self.batch_size,self.n_outputs))   # Inspired from the blogpost cited in report
        self.target_network_interval = 10000
        self.step_counter= 0
        self.iteration_loss = np.array([])


    def sample_experiences(self, batch_size):
        indices = np.random.randint(len(self.replay_memory), size=batch_size)
        batch = np.array([self.replay_memory[index] for index in indices])
        _states = np.take(batch, 0,axis=1)
        _actions = np.take(batch, 1,axis=1)
        _rewards = np.take(batch, 2,axis=1)
        _next_states = np.take(batch, 3,axis=1)
        _dones = np.take(batch, 4,axis=1)

        states = np.stack(_states, axis=0)
        actions = np.stack(_actions, axis=0)
        rewards = np.stack(_rewards, axis=0)
        next_states = np.stack(_next_states, axis=0)
        dones = np.stack(_dones, axis=0)

        return states, actions, rewards, next_states, dones


    
    def play_one_step(self, env, state, epsilon, step):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, info = env.step(action)

        return next_state, reward, done, info, action


    def train_control_plane(self, state, action, reward, terminal, observation, step, episode):
        next_state = np.append(state[:, :, 1:], observation, axis=2)
        # Change all positive rewards at 1 and all negative rewards at -1.
        reward = np.clip(reward, -1, 1)

        # Store data in the replay memory which compromising the size of it by removing the oldest element.
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) + 1>= NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        # Training based on training interval. 
        if episode > self.training_threshold and (step % TRAIN_INTERVAL == 0):
            self.train_one_step()

        # Updating the target network every 10000 steps to take advantage of Doubling DQN. 
        if self.step_counter % self.target_network_interval == 0:
            self.target_network.set_weights(self.q_network.get_weights())
        self.step_counter += 1
        return next_state



    def build_network(self):
        # Initializing network, with two inputs.
        input_frame = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, 4))
        action_one_hot = Input(shape=(4,))
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_frame)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
        flat_feature = Flatten()(conv3)
        hidden_feature = Dense(512)(flat_feature)
        lrelu_feature = LeakyReLU()(hidden_feature)
        q_value_prediction = Dense(4)(lrelu_feature)

        # Duelling network. Inspired by https://github.com/ShanHaoYu/Deep-Q-Network-Breakout
        hidden_feature_2 = Dense(512,activation='relu')(flat_feature)
        state_value_prediction = Dense(1)(hidden_feature_2)
        q_value_prediction = Lambda(lambda x: x[0]-K.mean(x[0])+x[1],output_shape=(self.n_outputs,))([q_value_prediction, state_value_prediction])

        # Doubling DQN. Estimating the best action with target network
        select_q_value_of_action = Multiply()([q_value_prediction,action_one_hot])
        target_q_value = Lambda(lambda x:K.sum(x, axis=-1, keepdims=True),output_shape=lambda_out_shape)(select_q_value_of_action)

        model = Model(inputs=[input_frame,action_one_hot], outputs=[q_value_prediction, target_q_value])
        
        # We update the target network only.
        model.compile(loss=['mse','mse'], loss_weights=[0.0,1.0],optimizer=self.optimizer)

        # Tensorboard loogging.
        self.tensorboard.set_model(model)
        return model        


    def train_one_step(self):
        
        # Sample a batch of data points from the replay buffer.
        experiences = self.sample_experiences(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = experiences
        terminal_batch = np.array(terminal_batch) + 0

        # Q value from target network
        target_q_values_batch = self.target_network.predict([next_state_batch,self.dummy_batch])[0]

        y_batch = reward_batch + (1 - terminal_batch) * self.gamma * np.max(target_q_values_batch, axis=-1)
        
        a_one_hot = np.zeros((self.batch_size,self.n_outputs))
        for idx,ac in enumerate(action_batch):
            a_one_hot[idx,ac] = 1.0
        # Perform one gradient step on the Q-Network
        loss = self.q_network.train_on_batch([state_batch,a_one_hot],[self.dummy_batch,y_batch])
        self.iteration_loss = loss[1]
        self.total_loss += loss[1]


    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_outputs)
        else:
            Q_values = self.q_network.predict([np.expand_dims(state,axis=0),self.dummy_input])
            toP = np.argmax(Q_values[0])
            return toP


    def train(self):
        best_score = self.best_score
        for episode in range(START_FROM, self.episods):
            game_score = 0
            step = 0
            terminal = False
            observation = self.env.reset()
            state = intial_state_preprocess(observation)
            all_actions = [] # Storing all action of the agent at each step for logging.
            while not terminal:
                # Epsilon decay
                epsilon = max(1 - episode / (500), 0.01)
                last_observation = observation
                # Playing one step.
                observation, reward, terminal, _, action = self.play_one_step(self.env, state, epsilon, step)
                processed_observation = state_preprocess(observation, last_observation)
                # Storing in replay buffer and performing training based on interval.
                state = self.train_control_plane(state, action, reward, terminal, processed_observation, step, episode)

                # Loggins params
                game_score += reward
                step +=1
                all_actions.append(action)
            self.tensorboard.on_epoch_end(episode, {'game_score': game_score, 'step': step, 'all_actions': all_actions, 'iteration_loss': np.mean(self.iteration_loss), 'total_loss': self.total_loss })
            self.iteration_loss = np.array([])
            # Saving model weights iteratively.
            if game_score > best_score:
                best_score = game_score
                self.q_network.save_weights('./weights/weights')
                self.target_network.save_weights('./weights/target_weights')
            print("\rGameScore: {}, Episode: {}, Steps: {}, eps: {:.3f}, Best_score: {}, Replay_Memery_len: {} ".format(game_score, episode, step + 1, epsilon, best_score, len(self.replay_memory)), end="") # Not shown
        
        self.q_network.save_weights('./weights/final_weights')
        self.target_network.save_weights('./weights/final_target_weights')
        


if __name__ == "__main__":
    agent = Agent()
    agent.train()

    
