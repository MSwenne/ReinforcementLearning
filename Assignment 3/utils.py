#####################################################################
#                                                                   #
# Reinforcement Learning Assignment 3: Function Approximation       #
#   utility functions                                               #
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
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "rl"
from skimage.transform import resize

import matplotlib as mpl
import matplotlib.pyplot as plt
import os
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)
import numpy as np
import cv2

import matplotlib.animation as animation
import matplotlib.pyplot as plt

def get_input(message, valid, ending="\n"):
    print(message, end=ending)
    result = input()
    while result not in valid:
        print("Invalid value!")
        print(message, end=ending)
        result = input()
    return result

def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch

def plot_animation(frames, repeat=False, interval=40):
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    anim = animation.FuncAnimation(
        fig, update_scene, fargs=(frames, patch),
        frames=len(frames), repeat=repeat, interval=interval)
    plt.close()
    return anim




def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def plot_environment(env, figsize=(5,4)):
    plt.figure(figsize=figsize)
    img = env.render(mode="rgb_array")
    plt.imshow(img)
    plt.axis("off")
    return img


def process(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)

    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)

def make_gray(img):

    # new_img = tf.image.rgb_to_grayscale(tf.convert_to_tensor(img), name=None)
    # new_img = tf.image.per_image_standardization(new_img)
    # new_img = resize(new_img, (84, 84))
    new_img = process(img)
    return new_img/255


def lambda_out_shape(input_shape):
    shape = list(input_shape)
    shape[-1] = 1
    return tuple(shape)

def list2np(in_list):
    return np.float32(np.array(in_list))