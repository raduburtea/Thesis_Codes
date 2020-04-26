import gym

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

# import tensorflow.contrib.slim as slim

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
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#Used for feature extraction in order to understand whether the states are the same
vector_size = 10*10 + 7 + 4

def create_cnn_vectorization():
    model = Sequential()
    
    model.add(Dense(512, input_shape=(10*10 + 7 + 4,)))
    model.add(Activation('relu'))

    model.add(Dense(4, init = 'lecun_uniform'))
    model.add(Activation('relu'))
    
    model.compile(loss='mse', optimizer='sgd')

    return model

class Model:
    def __init__(self, env, actions, actions_dict, gamma):
        self.env = env
        self.model = create_cnn_vectorization() #tracks the actual prediction
        self.actions = actions
        self.actions_dict = actions_dict
        self.gamma = gamma
    
    def predict(self, s, vector_size):
        return self.model.predict(s.reshape(-1, vector_size), verbose=0)[0]

    def update(self, s, Q):
        self.model.fit(s, Q, verbose = 0)

    def act(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([0,1,2,3], 1, p = [0.3,0.3,0.1,0.3])[0]
        return np.argmax(self.model.predict(state.reshape(-1, vector_size), verbose=0)[0])

    def copy(self):
        return self.model

    def update_weights(self, rewards, models, share_a, share_b):
        indexes_top_3 = sorted(range(len(rewards)), key=lambda i: rewards[i])[-3:][::-1]
        target_weights = self.model.get_weights()
        weights = [models[i].get_weights() for i in indexes_top_3]
        for i in range(len(target_weights)):
                target_weights[i] = share_a*weights[0][i] + share_b*weights[1][i] + (1-share_a-share_b)*weights[2][i]
        average_reward = share_a*rewards[indexes_top_3[0]] + share_b*rewards[indexes_top_3[1]] + (1-share_a-share_b)*rewards[indexes_top_3[2]]
        self.model.set_weights(target_weights)
        return average_reward

    def update(self, s, G):
        self.model.fit(s, np.array(G), nb_epoch=1, verbose=0)
                   
def transform(s):
    # We will crop the digits in the lower right corner, as they yield little 
    # information to our agent, as well as grayscale the frames.
    bottom_black_bar = s[84:, 12:]
    img = cv2.cvtColor(bottom_black_bar, cv2.COLOR_RGB2GRAY)
    bottom_black_bar_bw = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
    bottom_black_bar_b2 = cv2.resize(bottom_black_bar_bw, (84, 12), interpolation=cv2.INTER_NEAREST)
    
    # We will crop the sides of the screen, so we have an 84x84 frame, and grayscale them:
    upper_field = s[:84, 6:90]
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation=cv2.INTER_NEAREST)
    upper_field_bw = upper_field_bw.astype('float')/255
    
    # The car occupies a very small space, we do the same preprocessing:
    car_field = s[66:78, 43:53]
    img = cv2.cvtColor(car_field, cv2.COLOR_RGB2GRAY)
    car_field_bw = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY)[1]
    car_field_t = [car_field_bw[:, 3].mean()/255, 
                   car_field_bw[:, 4].mean()/255,
                   car_field_bw[:, 5].mean()/255, 
                   car_field_bw[:, 6].mean()/255]
    
    return bottom_black_bar_bw, upper_field_bw, car_field_t


def compute_steering_speed_gyro_abs(a):
    right_steering = a[6, 36:46].mean()/255
    left_steering = a[6, 26:36].mean()/255
    steering = (right_steering - left_steering + 1.0)/2
    
    left_gyro = a[6, 46:60].mean()/255
    right_gyro = a[6, 60:76].mean()/255
    gyro = (right_gyro - left_gyro + 1.0)/2
    
    speed = a[:, 0][:-2].mean()/255
    abs1 = a[:, 6][:-2].mean()/255
    abs2 = a[:, 8][:-2].mean()/255
    abs3 = a[:, 10][:-2].mean()/255
    abs4 = a[:, 12][:-2].mean()/255
    
    return [steering, speed, gyro, abs1, abs2, abs3, abs4]

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
actions_dict = {'left':[-0.8,0,0], 'right':[0.8,0,0], 'brake':[0,0,0.8], 'acc':[0,1,0]}

actions = ['left', 'right', 'brake', 'acc']



def play(agent, actions, actions_dict, epsilon, species, share_a, share_b):
    models = []
    rewards = []
    print(epsilon)
    iter = 1
    for i in range(species):
        observation = env.reset()
        if i>=4:
            new_model = Model(env, actions, actions_dict, 0.8)
            model_gen = new_model.model
            models.append(model_gen)
        else:
        
            model_gen = model.copy()
            models.append(model_gen)
        totalreward = 0
        iter = 1
        done = False
        
        while not done:
            env.render()
            a, b, c = transform(observation)
            # state = state_intermed.reshape(-1, 84, 84, 1)
           
            state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(),
                               b.reshape(1, -1).flatten(),c), axis=0)
            action = agent.act(state, epsilon)
            observation, reward, done, info = env.step(actions_dict[actions[action]])
            a,b,c = transform(observation)
            new_state = np.concatenate((np.array([compute_steering_speed_gyro_abs(a)]).reshape(1,-1).flatten(),
                               b.reshape(1, -1).flatten(),c), axis=0)
            qval = model_gen.predict(new_state.reshape(-1, vector_size), 10*10 + 7 + 4)[0]
            G = reward + 0.99*np.max(qval)
            y = qval[:]
#            print('it is:', np.amax(qval))
            y[np.argmax(qval)] = G
            model_gen.fit(state.reshape(-1, vector_size), y.reshape(1,-1), epochs = 1, verbose = 0)
            totalreward+=reward
            state = new_state
        rewards.append(totalreward)

    iter += 1
    weighted_average = agent.update_weights(rewards, models, share_a, share_b)
    return weighted_average, iter

    


N=200
totalrewards = np.empty(N)
costs = np.empty(N)
gamma = 0.8
model = Model(env, actions, actions_dict, gamma)
actions = ['left', 'right', 'brake', 'acc']
species = 7
share_a = 0.6
share_b = 0.2
env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
for n in range(1,N):
    eps = 1 / np.sqrt(n+5)
    totalreward, iters = play(model, actions, actions_dict, eps, species, share_a, share_b)

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

