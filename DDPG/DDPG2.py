import gym
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy.misc as smp
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from gym.envs.registration import registry, register, make, spec
# from keras.applications import imagenet_utils

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
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
from keras.optimizers import Adamax
from collections import deque
import cv2
from skimage import color, transform
import gym
from gym import wrappers

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
import tensorflow as tf
from ActorNetwork23 import ActorNetwork
from CriticNetwork23 import CriticNetwork

# from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


class Model:
    def __init__(self, env, gamma, sess):
        self.env = env
        self.sess = sess
        np.random.seed(0)
        self.actor = ActorNetwork(self.sess, (84,84,4), 3, 0.001, 0.0001)
        self.critic = CriticNetwork(self.sess, ( 84,84,4), 3, 0.001, 0.0001)
        self.memory = deque(maxlen=4000)
        self.gamma = gamma
        
    # def predict(self, s):
    #     return self.model.predict(s)[0]

    # def update(self, s, Q):
    #     self.model.fit(s, Q, verbose = 0)

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 8
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)

        states = np.asarray([e[0] for e in samples])
        actions = np.asarray([e[1] for e in samples])
        rewards = np.asarray([e[2] for e in samples])
        new_states = np.asarray([e[3] for e in samples])
        dones = np.asarray([e[4] for e in samples])
        y_t = np.asarray([e[1] for e in samples])

        target_q_values = [self.critic.target_model.predict([state, self.actor.target_model.predict(state)]) for state in new_states]  
       
        for k in range(8):
            if dones[k]:
                y_t[k] = rewards[k]
            else:
                y_t[k] = rewards[k] + 0.8*target_q_values[k]
   
        
        # loss += self.critic.model.train_on_batch([states,actions], y_t) 
        a_for_grad = [self.actor.model.predict(state) for state in states]
        grads = [self.critic.gradients(states[i], a_for_grad[i]) for i in range(8)] 
        [self.actor.train(states[i], grads[i]) for i in range(8)]
        self.actor.target_train()
        self.critic.target_train()



def rgb2gray(rgb):
    i = rgb[:84, 5:89, :]
    # image = np.round(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]))
    # return image.reshape((84, 84))
    i = 2 * color.rgb2gray(i) - 1
    return i.reshape((84, 84))

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



def playGame():    #1 means Train, 0 means simply Run
    GAMMA = 0.99
    TAU = 0.99     #Target Network HyperParameters
    LRA = 0.01    #Learning rate for Actor
    LRC = 0.01     #Lerning rate for Critic
    #of sensors input

    np.random.seed(1337)

    EXPLORE = 100000.
    episode_count = 600
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    indicator = 0
    rew =[]
    env = gym.make('CarRacing-v1')
    env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    agent = Model(env,  0.85, sess)
    
    #Tensorflow GPU optimization - pe asta o am

    # actor = agent.actor
    # critic = agent.critic
    
    # Generate a Torcs environment
    
        
        #Now load the weight
    for i in range(episode_count): 
        if i <=10:
            eps = 1
        else:
            eps = 1/np.sqrt(n-2)
        observation = env.reset()
        total_reward = 0
        images = ImageMemory()
        done = False
        while not done:
            env.render()
            state_intermed = rgb2gray(observation)
            images.add_image(state_intermed)
            state = images.get_stacked_images().reshape(-1, 84, 84, 4)
            if np.random.random()>eps:
                action = agent.actor.model.predict(state)[0]
            else:
                action = env.action_space.sample()
            # print(action)
            observation, reward, done, info = env.step(action)

            new_state_intermed = rgb2gray(observation)
            images.add_image(new_state_intermed)
            new_state = images.get_stacked_images().reshape(-1, 84, 84, 4)

        
            agent.remember(state, action, reward, new_state, done)     #Add replay buffer
            
            #Do the batch update
            agent.replay()

            total_reward += reward
            state = new_state
        
            # print("Episode", i, "Step", step, "Action", a_t, "Reward", r_t, "Loss", loss)
        
            step += 1
        rew.append(total_reward)
        print("TOTAL REWARD @ " + str(i) +"-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    if i%50 == 0:
            self.agent.model.save_weigths("actormodel.h5", overwrite = True)
            self.critic.model.save_weigths("criticmodel.h5", overwrite = True) 

    env.close()  # This is for shutting down TORCS
    return rew
    print("Finish.")

rew = playGame()
plt.plot(rew)
plt.show()

