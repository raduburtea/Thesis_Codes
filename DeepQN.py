import gym
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.misc as smp
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from gym.envs.registration import registry, register, make, spec
# from keras.applications import imagenet_utils

register(
    id='CarRacing-v1', # CHANGED
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=1200, # CHANGED
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


# from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


def create_cnn_vectorization():
    model = Sequential()
    model.add(Conv2D(filters = 16, kernel_size = 8, strides = (3,3), input_shape=( 84, 84, 4)))
    model.add(Activation('relu'))
    model.add(Conv2D(filters = 32, kernel_size = 4,  input_shape=( 84, 84, 4)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(256))  # 7x7+3 or 14x14+3
    model.add(Activation('relu'))
    
    model.add(Dense(15))
    model.add(Activation('relu'))  # linear output so we can have a range of real-valued opts.
    
    model.compile(loss='mse', optimizer=Adamax(lr=0.001))  # lr=0.001
    
    
    return model


class Model:
    def __init__(self, env, actions, actions_dict, gamma):
        self.env = env
        self.model = create_cnn_vectorization() #tracks the actual prediction
        self.target_model = create_cnn_vectorization() #tracks the action we want the agent to take
        self.memory = deque(maxlen=2000)
        self.actions = actions
        self.actions_dict = actions_dict
        self.gamma = gamma
        
    def predict(self, s):
        return self.model.predict(s)[0]

    def update(self, s, Q):
        self.model.fit(s, Q, verbose = 0)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 8
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            
            if done:
                target[0][action] = reward
            else:
                Q_future = max(
                    self.target_model.predict(new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(state, target, epochs=1, verbose=0)

    def act(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice([i for i in range(len(actions))], 1)[0]
        return np.argmax(self.model.predict(state)[0])

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)



def rgb2gray(rgb):
    i = rgb[:84, 5:89, :]
    i = tf.image.rgb_to_grayscale(i)
    i = tf.reshape(i, [84, 84])
    return i

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
        # if len(self.images) <2:
        #     self.images.append(image)
        # else:
            self.images.pop(0)
            self.images.append(image)

    def get_stacked_images(self):
        return np.stack(self.images, axis = 2)

    
    def print_images(self):
        print(self.images)




#due to the high dimensionality of the action space I have decided to make it discrete in order to implement a q-learning algorithm


def action_epsilon_greedy(state, epsilon):
    flag = 'random'
    qval = model.predict(state)
    # print('qval is', qval)
    if np.random.random() > epsilon:
        # print(state)
        flag='not'
        return self.actions[np.argmax(qval)]
    else:
        return np.random.choice([0,1,2,3], 1)[0]

def action_creation(output, actions, actions_dict, flag):
    # if flag == 'random':
        return actions_dict[actions[output]]


env.action_space.sample()

def play(agent, actions, actions_dict, epsilon):
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

        action = agent.act(state, epsilon)
        observation, reward, done, info = env.step(actions[action])
        new_state_intermed = rgb2gray(observation)
        images.add_image(new_state_intermed)
        
        new_state = images.get_stacked_images().reshape(-1, 84, 84, 4)
        # new_state = new_state_intermed.reshape(-1, 84, 84, 1)
        agent.remember(state, action, reward, new_state, done)

        agent.replay()
        agent.target_train()
        # print('qval is ', qval_prime)
        # state2.show_class()
        state = new_state


        totalreward+=reward
        iter+=1

    return totalreward, iter

actions = [[-0.4,0.3,0], [0.4,0.3,0], [0.0,0.3,0],  [0,0.2,0],  [0,0,0], [-0.1,0.1,0], [0.1,0.1,0],  [-0.2,0.2,0], [0.2,0.2,0], [-0.6,0.4,0.1], [0.6,0.4,0.1], [-0.8,0.3,0.1], [0.8,0.3,0.1] , [-0.4,0.2,0], [0.4,0.2,0]]

# with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
N=300
totalrewards = np.empty(N)
costs = np.empty(N)
agent = Model(env, actions, actions_dict, 0.85)
# actions = ['left', 'right', 'brake', 'acc',]
env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
for n in range(1,N):
    if n <= 1:
        eps = 1
    elif eps <= 0.15:
        eps = 0.15
    else:
        eps = 1 / np.sqrt(n)
    totalreward, iters = play(agent, actions, actions_dict, eps)

    totalrewards[n] = totalreward
    print("Episode: ", n,
          ", iters: ", iters,
          ", total reward: ", totalreward,
          ", epsilon: ", eps,
          ", average reward (of last 100): ", totalrewards[max(0,n-100):(n+1)].mean()
         )
# We save the model every 10 episodes:
    if n%10 == 0:
        agent.model.save('race-car_larger2.h5')
 
env.close()
plot_running_avg(totalrewards)
