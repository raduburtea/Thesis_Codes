import gym
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
# from keras.applications.mobilenet import MobileNet

# from rl.agents.dqn import DQNAgent
# from rl.policy import EpsGreedyQPolicy
# from rl.memory import SequentialMemory
# from keras.applications import VGG16
import scipy.misc as smp
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from gym.envs.registration import registry, register, make, spec
# from keras.applications import imagenet_utils

register(
    id='CarRacing-v1', # CHANGED
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=1500, # CHANGED
    reward_threshold=900,
)

# import tensorflow.contrib.slim as slim

import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adamax
from collections import deque
import cv2
from skimage import color, transform
import gym
from gym import wrappers
env = gym.make('CarRacing-v1')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import tensorflow as tf



def create_cnn_vectorization():
    # if os.path.exists('race-car_larger2.h5'):
    #     print("Model is loaded")
    #     return load_model('race-car_larger2.h5')

    model = Sequential()

    model.add(Conv2D(filters = 16, kernel_size = 8, strides = (4,4), input_shape=( 84, 84, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 32, kernel_size = 4,  input_shape=( 84, 84, 4)))
    model.add(Activation('relu'))
    model.add(Flatten())

    
    model.add(Dense(256, init='lecun_uniform'))
    model.add(Activation('relu'))

    model.add(Dense(128, init='lecun_uniform'))
    model.add(Activation('relu'))

    model.add(Dense(11, init = 'lecun_uniform'))
    model.add(Activation('linear'))
    
    model.compile(loss='mse', optimizer=Adamax(lr=0.001))  # lr=0.001
    
    
    return model


class Model:
    def __init__(self, env, gamma):
        self.env = env
        self.model = create_cnn_vectorization() #tracks the actual prediction
        self.target_model = create_cnn_vectorization() #tracks the action we want the agent to take
        self.memory = deque(maxlen=2000)

        self.gamma = gamma
        
    def predict(self, s):
        return self.model.predict(s, verbose=0)[0]

    def update(self, s, Q):
        self.model.fit(s, Q, verbose = 0)

    def remember(self, state, action, arg, reward, new_state, done):
        self.memory.append([state, action, arg, reward, new_state, done])

    def replay(self):
        batch_size = 8
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, arg, reward, new_state, done = sample
            target = self.target_model.predict(state, verbose=0)
            
            if done:
                target[0][arg] = reward
            else:
                Q_future = max(
                    self.target_model.predict(new_state, verbose=0)[0])
                target[0][arg] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def act(self, state, epsilon):
        if np.random.random() < epsilon:
            return convert_argmax_qval_to_env_action(np.random.choice([i for i in range(11)], 1)[0]), np.random.choice([i for i in range(11)], 1)[0]
        return convert_argmax_qval_to_env_action(np.argmax(self.model.predict(state, verbose=0)[0])), np.argmax(self.model.predict(state, verbose=0)[0]) 

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = target_weights[i]*0.3 + 0.7*weights[i]
        self.target_model.set_weights(target_weights)



def rgb2gray(rgb):
    i = rgb[:84, 5:89, :]
    i = 2 * color.rgb2gray(i) - 1
    return i.reshape((84, 84))

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


class ImageMemory:
    def __init__(self):
        self.images = [np.zeros((84,84)) for i in range(4)]

    def add_image(self, image):

            self.images.pop(0)
            self.images.append(image)

    def get_stacked_images(self):
        return np.stack(self.images, axis = 2)

    
    def print_images(self):
        print(self.images)


def convert_argmax_qval_to_env_action(output_value):
    # We reduce the action space to 
    
    gas = 0.0
    brake = 0.0
    steering = 0.0
    
    # Output value ranges from 0 to 10:
    
    if output_value <= 8:
        # Steering, brake, and gas are zero
        output_value -= 4
        steering = float(output_value)/4
    elif output_value >=9 and output_value <=9:
        output_value -= 8
        gas = float(output_value)/3  # 33% of gas
    elif output_value >= 10 and output_value <= 10:
        output_value -= 9
        brake = float(output_value)/2  # 50% of brake
    else:
        print("Error")  #Why?
    
    return [steering, gas, brake]

def play(agent,  epsilon):
    print(epsilon)
    observation = env.reset()
    totalreward = 0
    iter = 1
    done = False
    # state = rgb2gray(observation)
    images = ImageMemory()
    while not done:
        env.render()
        
        state_intermed = rgb2gray(observation)
        images.add_image(state_intermed)
        state = images.get_stacked_images().reshape(-1, 84, 84, 4)


        action, argmax = agent.act(state, epsilon)

        observation, reward, done, info = env.step(action)
        new_state_intermed = rgb2gray(observation)
        
        images.add_image(new_state_intermed)
        
        new_state = images.get_stacked_images().reshape(-1, 84, 84, 4)
        # new_state = new_state_intermed.reshape(-1, 84, 84, 1)
        agent.remember(state, action, argmax, reward, new_state, done)

        agent.replay()
        agent.target_train()
        # print('qval is ', qval_prime)
        # state2.show_class()
        state = new_state
        # Q = reward+0.99*(np.max(qval_prime))
    
        totalreward+=reward
        iter+=1

    return totalreward, iter
# with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
N=400
totalrewards = np.empty(N)
costs = np.empty(N)
agent = Model(env,  0.99)
# actions = ['left', 'right', 'brake', 'acc',]
eps = 1
env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
for n in range(1,N):
    if n <= 10:
        eps -= 0.05
    else:
        eps = 1/np.sqrt(n)
    if eps <= 0.05:
        eps = 0.05
    totalreward, iters = play(agent,  eps)

    totalrewards[n] = totalreward
    print("Episode: ", n,
          ", iters: ", iters,
          ", total reward: ", totalreward,
          ", epsilon: ", eps,
          ", average reward (of last 100): ", totalrewards[max(0,n-99):(n)].mean()
         )
# We save the model every 10 episodes:
    if n%50 == 0:
        agent.model.save('race-car_larger2.h5')
    if n % 50 == 0:
        with open('Rew.txt', 'w') as filehandle:
            for listitem in totalrewards:
                filehandle.write('%s\n' % listitem)
env.close()
plot_running_avg(totalrewards)

