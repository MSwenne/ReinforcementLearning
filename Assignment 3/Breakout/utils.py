import tensorflow as tf
import numpy as np
import cv2
from skimage.transform import resize
import os

FRAME_WIDTH = 84
FRAME_HEIGHT = 84   
STACK_SIZE = 4

def process(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)

    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

def make_gray(img):
    new_img = process(img)
    return new_img/255


# Inspired by the blog, cited in report.
def lambda_out_shape(input_shape):
    shape = list(input_shape)
    shape[-1] = 1
    return tuple(shape)


def intial_state_preprocess(observation):
    processed_observation = make_gray(observation)
    proccessed_state = [processed_observation for _ in range(STACK_SIZE)]
    return np.stack(proccessed_state, axis=2).reshape(FRAME_WIDTH, FRAME_HEIGHT, STACK_SIZE)

def state_preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = make_gray(processed_observation)
    return np.reshape(processed_observation, (FRAME_HEIGHT, FRAME_WIDTH, 1))
