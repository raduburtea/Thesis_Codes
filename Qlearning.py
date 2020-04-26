import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications.mobilenet import MobileNet

# from rl.agents.dqn import DQNAgent
# from rl.policy import EpsGreedyQPolicy
# from rl.memory import SequentialMemory
from keras.applications import VGG16
import scipy.misc as smp
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from gym.envs.registration import registry, register, make, spec
from keras.applications import imagenet_utils

register(
    id='CarRacing-v1', # CHANGED
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=1600, # CHANGED
    reward_threshold=900,
)

import tensorflow.contrib.slim as slim

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adamax

import cv2

import gym
from gym import wrappers
env = gym.make('CarRacing-v1')

#Used for feature extraction in order to understand whether the states are the same
def create_cnn_vectorization():
    model = Sequential()
    model.add(Conv2D(16, (8, 8), activation='relu', input_shape=(78, 78, 1)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Conv2D(16, (8, 8), activation='relu', input_shape=(78, 78, 1)))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Flatten())
    return model


# model = VGG16(weights="imagenet", include_top=False)

def image_vectorization(image, pre_model):
    # image = np.expand_dims(image, axis=0)
    # image = imagenet_utils.preprocess_input(image)
    # print(image)
    image = image.reshape((-1, 78, 78, 3))
    features = pre_model.predict(image, batch_size=32)
    features_flatten = features.reshape((features.shape[0], 7 * 7 * 512))
    return features_flatten

def rgb2gray(rgb):
    rgb = rgb[:78, 8:86, :]
    image = np.round(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]))
    return image.reshape((-1, 78, 78, 1))

def transform(image, model):
    return np.round(model.predict(rgb2gray(image))[0], 0)


def check_state(out, out2):
    if np.sum(out2 == out) == len(out):
        return True
    else:
        return False

# if np.sum(out2 == out) == len(out[0]):
#     print('Smash')

# class States:
#     def __init__(self):
#         states = []
#
#     def add_state(self, state):
#         self.states.append(state)
#
#     def check_if_in(self, state_new):
#         for state in self.states:
#             if not check_state(state, state_new):
#                 return False
#         return True


def plot_running_avg(total_rewards):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()




#due to the high dimensionality of the action space I have decided to make it discrete in order to implement a q-learning algorithm


brake = [0,0,0.8]
acc = [0,1,0.8]
right = [1,1,0.8]
left = [-1,1,0.8]
# nothing = [0,0,0]
actions_dict = {'left':[-0.8,0,0], 'right':[0.8,0,0], 'brake':[0,0,0.8], 'acc':[0,0.1,0]}

actions = ['left', 'right', 'brake', 'acc']

def return_all_max(actions):
    listOfKeys = list()
    itemMaxValue = max([(value, key) for key, value in actions.items()])[0]
    # Iterate over all the items in dictionary to find keys with max value
    for key, value in actions.items():
        if value == itemMaxValue:
            listOfKeys.append(key)
    # print(np.random.choice(listOfKeys, 1)[0])
    return np.random.choice(listOfKeys, 1)[0]


class state_action:

    def __init__(self, state, actions):
        self.state = state
        self.actions_vals = {act:0 for act in actions}

    def add_action(self, action, qval): #update qval? - I think so
        if self.actions_vals[action] < qval:
            self.actions_vals[action] = qval

    def show_class(self):
        print(self.actions_vals)

    def return_max_act(self):
            return return_all_max(self.actions_vals)

    def update_qval(self, action, qval_new):
        self.actions_vals[action] = qval_new

    def qval(self, action):
        try:
            return self.actions_vals[action]
        except:
            print('Stg is wrong')

    def check_empty(self):
        if len(self.qvalues) == 0:
            return True

def action_epsilon_greedy(state, epsilon):
    if np.random.random() > epsilon:
        # print(state)
        return state.return_max_act()
    else:
        return np.random.choice(actions, 1, p = [0.30, 0.30, 0.1, 0.3])[0]

def check_state(out, out2):
    if np.sum(out2 == out) == len(out):
        return True
    else:
        return False

def check_if_in(states, state_new):
    if type(state_new).__name__ != 'state_action':
        print('Prblm here')
    for i in range(len(states)):
        if check_state(states[i].state, state_new):
            # print('Sure?')
            return states[i]
    return False

def appending_states(states, state):
    if check_if_in(states, state)==False:
        # print('This')
        states.append(state)
    else:

        state = check_if_in(states, state)
    return state, states

env.action_space.sample()

def play(states, model, actions, actions_dict, epsilon):
    observation = env.reset()
    totalreward = 0
    iter = 1
    done = False
    state = state_action(transform(observation, model), actions)
    state, states = appending_states(states, state)
    action = action_epsilon_greedy(state, epsilon)
    while not done:
        env.render()

        # print('The action is:', action)
        observation, reward, done, info = env.step(np.array(actions_dict[action]).astype('float32'))

        state2 = state_action(transform(observation, model), actions)
        state2, states = appending_states(states, state)
        # state2.show_class()
        action_prime = action_epsilon_greedy(state2, epsilon)
        q_val = state.qval(action)
        q_val += 0.4*(reward+0.8*state2.qval(action_prime) - q_val)

        state.update_qval(action, q_val)

        state = state2
        action = action_prime
        # print('Prime', action_prime)

        totalreward+=reward
        iter+=1

    return totalreward, iter

N=200
totalrewards = np.empty(N)
costs = np.empty(N)
model = create_cnn_vectorization()
states = []
actions = ['left', 'right', 'brake', 'acc']
for n in range(1,N):
    eps = 1 / np.power(n, 1 / 4)
    totalreward, iters = play(states, model, actions, actions_dict, eps)

    totalrewards[n] = totalreward
    print("Episode: ", n,
          ", iters: ", iters,
          ", total reward: ", totalreward,
          ", epsilon: ", eps,
          ", average reward (of last 100): ", totalrewards[max(0,n-100):(n+1)].mean()
         )
    # We save the model every 10 episodes:
    if n%10 == 0:
        model.model.save('race-car_larger.h5')

env.close()




# observation = env.reset()
# print(observation)
# for i in range(200):
#     # observation = env.reset()
#     env.render()
#     action = [-1,0.4,0]
#     observation, reward, done, info = env.step(action)
#     # graying(observation)

env.close()
#
# #tests
# mine = state_action([1,2,3,4])
# mine2 = state_action([1,2,3,4])
# check_if_in(states, mine2.state)
