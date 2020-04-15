import gym
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
from collections import deque
import sys
import os
import numpy as np
import tensorflow as tf

# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)

# To get smooth animations
import matplotlib.animation as animation
# mpl.rc('animation', html='jshtml')

# # Where to save the figures
# PROJECT_ROOT_DIR = "."
# CHAPTER_ID = "rl"
# IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
# os.makedirs(IMAGES_PATH, exist_ok=True)

# try:
#     import pyvirtualdisplay
#     display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
# except ImportError:
#     pass

# def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
#     path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
#     print("Saving figure", fig_id)
#     if tight_layout:
#         plt.tight_layout()
#     plt.savefig(path, format=fig_extension, dpi=resolution)

# def plot_environment(env, figsize=(5,4)):
#     plt.figure(figsize=figsize)
#     img = env.render(mode="rgb_array")
#     plt.imshow(img)
#     plt.axis("off")
#     return img

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

batch_size = 32
discount_rate = 0.95
optimizer = keras.optimizers.Adam(lr=1e-3)
loss_fn = keras.losses.mean_squared_error
env = gym.make('MountainCar-v0')
env.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
rewards = [] 
best_score = -210
input_shape = [2] # == env.observation_space.shape
n_outputs = env.action_space.n

replay_memory = deque(maxlen=2000)

model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])


def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones


def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

iters = 100

def cal_reward(obs):
    pos, vel = obs
    max_pos = 0.7
    max_vel = 0.07
    return (1/(max_pos - pos)) * abs(vel) 

for episode in range(iters):
    obs = env.reset()    
    game_score = 0
    for step in range(200):
        epsilon = max(1 - episode / iters, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        
        game_score += cal_reward(obs)
        if done:
            break
    rewards.append(game_score)
    if game_score > best_score:
        best_weights = model.get_weights() # Not shown
        best_score = game_score # Not shown
    print("\rEpisode: {}, Steps: {}, eps: {:.3f}, Best_score: {} ".format(episode, step + 1, epsilon, best_score), end="") # Not shown
    if episode > 20:
        training_step(batch_size)

model.set_weights(best_weights)

state = env.reset()

frames = []

for step in range(200):
    action = epsilon_greedy_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        break
    img = env.render(mode="rgb_array")
    frames.append(img)

plot_animation(frames)
