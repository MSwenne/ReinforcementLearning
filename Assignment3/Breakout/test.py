#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 3: Function Approximation       #
#   Test Breakout trained model                                     #
#                                                                   #
# Made by:  Amin Moradi         S2588862                            #
#           Bartek Piaskowski   S2687194                            #
#           Martijn Swenne      S1923889                            #
#                                                                   #
# Last edited: 28 April 2020                                        #
# All rights reserved                                               #
#                                                                   #
#####################################################################

import gym
import sys
import os
import csv
import cv2
import functools
from collections import deque

from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model, load_model

import numpy as np
# Local
from utils import make_gray, intial_state_preprocess, state_preprocess


FRAME_WIDTH = 84
FRAME_HEIGHT = 84

def lambda_out_shape(input_shape):
    shape = list(input_shape)
    shape[-1] = 1
    return tuple(shape)


optimizer = keras.optimizers.Adam(lr=0.0001)
n_outputs = 4
np.random.seed(42)
tf.random.set_seed(42)
FRAME_WIDTH = 84
FRAME_HEIGHT = 84

def build_network():
    # Consturct model
    Conv2D = keras.layers.Conv2D
    Dense = keras.layers.Dense
    Input = keras.layers.Input
    Flatten = keras.layers.Flatten
    LeakyReLU = keras.layers.LeakyReLU
    Lambda = keras.layers.Lambda
    Multiply = keras.layers.Multiply

    input_frame = Input(shape=(FRAME_WIDTH, FRAME_HEIGHT, 4))
    action_one_hot = Input(shape=(4,))
    conv1 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')(input_frame)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')(conv1)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')(conv2)
    flat_feature = Flatten()(conv3)
    hidden_feature = Dense(512)(flat_feature)
    lrelu_feature = LeakyReLU()(hidden_feature)
    q_value_prediction = Dense(4)(lrelu_feature)

    hidden_feature_2 = Dense(512,activation='relu')(flat_feature)
    state_value_prediction = Dense(1)(hidden_feature_2)
    q_value_prediction = Lambda(lambda x: x[0]-K.mean(x[0])+x[1],output_shape=(n_outputs,))([q_value_prediction, state_value_prediction])
    select_q_value_of_action = Multiply()([q_value_prediction,action_one_hot])
    target_q_value = Lambda(lambda x:K.sum(x, axis=-1, keepdims=True),output_shape=lambda_out_shape)(select_q_value_of_action)

    model = Model(inputs=[input_frame,action_one_hot], outputs=[q_value_prediction, target_q_value])
    model.compile(loss=['mse','mse'], loss_weights=[0.0,1.0],optimizer=optimizer)
    return model   



if __name__ == "__main__":
    q_network = build_network()
    target_network = build_network()
    q_network.load_weights("./weights/weights")
    target_network.load_weights("./weights/target_weights")
    env = gym.make('Breakout-v0')
    env.seed(42)
    n_outputs =  env.action_space.n
    dummy_input = np.zeros((1,n_outputs))
    game_restults=[]

    for iteration in range(30):
        observation = env.reset()
        done = False
        game_score = 0
        next_state = intial_state_preprocess(observation)
        counter = 0
        while not done and counter < 6000:
            counter +=1
            Q_values = q_network.predict([np.expand_dims(next_state,axis=0),dummy_input])
            action = np.argmax(Q_values[0])
            state, reward, done, info = env.step(action)
            state = make_gray(state)
            next_state = np.append(next_state[:, :, 1:], state, axis=2)
            game_score += reward
            print("\rGameScore: {}, Episode: {} , Counter: {} ".format(game_score, iteration, counter), end="") # Not shown
        game_restults.append((iteration, game_score))
    # Save to CSV
    with open('last_30.csv','w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['game','score'])
        for row in game_restults:
            csv_out.writerow(row)