
import gym
from tensorflow import keras
from collections import deque
import sys
import numpy as np
import tensorflow as tf
import functools
from statistics import mean 

from tensorflow.keras.models import Sequential, Model, load_model
import tensorflow.keras.backend as K

from utils import make_gray, lambda_out_shape, list2np
import random

FRAME_WIDTH = 84
FRAME_HEIGHT = 84
TRAIN_INTERVAL = 4
NUM_REPLAY_MEMORY = 1000000

class Agent():
    def __init__(self):
        self.batch_size = 32
        self.tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir='./my_tf_logs',
            histogram_freq=0,
            batch_size=self.batch_size,
            write_graph=True,
            write_grads=True
        )
        self.total_loss = 0.0
        # self.iters = 100
        self.training_threshold = 20
        self.episods = 5000
        self.discount_rate = 0.95
        self.optimizer = keras.optimizers.Adam(lr=0.0001)
        self.loss_fn = keras.losses.mean_squared_error
        self.env = gym.make('Breakout-v0')
        self.env.seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        self.rewards = [] 
        self.best_score = -np.inf
        # self.input_shape = [2] # == env.observation_space.shape
        # n_outputs = env.action_space.n
        self.n_outputs =  self.env.action_space.n
        self.state_length = 4

        self.replay_memory = deque(maxlen=NUM_REPLAY_MEMORY)
        self.epsilon = 0

        self.q_network = self.build_network()
        self.target_network = self.build_network()
        
        self.dummy_input = np.zeros((1,self.n_outputs))
        self.dummy_batch = np.zeros((self.batch_size,self.n_outputs))
        self.target_update_interval = 1000
        self.t = 0

        self.iteration_loss = np.array([])
        # Set pretrained weights 
        # self.q_network.load_weights('./weights/weights')
        # self.target_network.set_weights(self.q_network.get_weights())

    def epsilon_greedy_policy(self, state, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(4)
        else:
            # Q_values = self.q_network.predict(np.stack([state], axis=0))
            Q_values = self.q_network.predict([np.expand_dims(state,axis=0),self.dummy_input])
            toP = np.argmax(Q_values[0])
            return toP

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


    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = make_gray(processed_observation)
        proccessed_state = [processed_observation for _ in range(self.state_length)]
        
        return np.stack(proccessed_state, axis=0).reshape(FRAME_WIDTH, FRAME_HEIGHT, 4)

    def preprocess(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = make_gray(processed_observation)
        return np.reshape(processed_observation, (FRAME_HEIGHT, FRAME_WIDTH, 1))



    def run(self, state, action, reward, terminal, observation, step, episode):
        next_state = np.append(state[:, :, 1:], observation, axis=2)
        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if episode > self.training_threshold and (step % TRAIN_INTERVAL == 0):
            self.train_network()

        if self.t % self.target_update_interval == 0:
            self.target_network.set_weights(self.q_network.get_weights())
        self.t += 1
        return next_state



    def build_network(self):
        # Consturct model
        Conv2D = keras.layers.Conv2D
        Dense = keras.layers.Dense
        Input = keras.layers.Input
        Flatten = keras.layers.Flatten
        LeakyReLU = keras.layers.LeakyReLU
        Lambda = keras.layers.Lambda
        Multiply = keras.layers.Multiply
        BatchNormalization = keras.layers.BatchNormalization
    
        input_frame = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, 4))
        action_one_hot = Input(shape=(4,))
        conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_frame)
        # normed1 = BatchNormalization()(conv1)
        conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
        # normed2 = BatchNormalization()(conv2)
        conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
        # normed3 = BatchNormalization()(conv3)
        flat_feature = Flatten()(conv3)
        hidden_feature = Dense(512)(flat_feature)
        lrelu_feature = LeakyReLU()(hidden_feature)
        q_value_prediction = Dense(4)(lrelu_feature)

        
        hidden_feature_2 = Dense(512,activation='relu')(flat_feature)
        state_value_prediction = Dense(1)(hidden_feature_2)
        q_value_prediction = Lambda(lambda x: x[0]-K.mean(x[0])+x[1],output_shape=(self.n_outputs,))([q_value_prediction, state_value_prediction])



        select_q_value_of_action = Multiply()([q_value_prediction,action_one_hot])
        target_q_value = Lambda(lambda x:K.sum(x, axis=-1, keepdims=True),output_shape=lambda_out_shape)(select_q_value_of_action)

        model = Model(inputs=[input_frame,action_one_hot], outputs=[q_value_prediction, target_q_value])
        
        # MSE loss on target_q_value only
        model.compile(loss=['mse','mse'], loss_weights=[0.0,1.0],optimizer=self.optimizer)

        self.tensorboard.set_model(model)
        return model        


    def train_network(self):
        y_batch = []
        experiences = self.sample_experiences(self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = experiences
        
        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0
        # Q value from target network
        target_q_values_batch = self.target_network.predict([list2np(next_state_batch),self.dummy_batch])[0]

        y_batch = reward_batch + (1 - terminal_batch) * self.discount_rate * np.max(target_q_values_batch, axis=-1)
        
        a_one_hot = np.zeros((self.batch_size,self.n_outputs))
        for idx,ac in enumerate(action_batch):
            a_one_hot[idx,ac] = 1.0

        loss = self.q_network.train_on_batch([list2np(state_batch),a_one_hot],[self.dummy_batch,y_batch])
        self.iteration_loss = loss[1]
        self.total_loss += loss[1]


    def exec(self):
        best_score = -np.inf
        for episode in range(self.episods):
            game_score = 0
            step = 0
            terminal = False
            observation = self.env.reset()
            for _ in range(np.random.randint(30)):
                last_observation = observation
                observation, _, _, _ = self.env.step(0)  # Do nothing
            state = self.get_initial_state(observation, last_observation)
            all_actions = []
            while not terminal:
                last_observation = observation
                step +=1
                epsilon = max(1 - episode / 500, 0.01)
                observation, reward, terminal, info, action = self.play_one_step(self.env, state, epsilon, step)
                # print("action:" ,action)
                game_score += reward
                processed_observation = self.preprocess(observation, last_observation)
                state = self.run(state, action, reward, terminal, processed_observation, step, episode)
                all_actions.append(action)
            self.tensorboard.on_epoch_end(episode, {'game_score': game_score, 'step': step, 'all_actions': all_actions, 'iteration_loss': np.mean(self.iteration_loss), 'total_loss': self.total_loss })
            self.iteration_loss = np.array([])
            if game_score > best_score:
                best_score = game_score # Not shown
                self.q_network.save_weights('./weights/weights')
                self.target_network.save_weights('./weights/target_weights')
            # self.q_network.summary()
            print("\rGameScore: {}, Episode: {}, Steps: {}, eps: {:.3f}, Best_score: {} ".format(game_score, episode, step + 1, epsilon, best_score), end="") # Not shown
        
        self.q_network.save_weights('./final_weights/weights')
        self.target_network.save_weights('./final_weights/target_weights')
        


if __name__ == "__main__":
    agent = Agent()
    agent.exec()

    
