import numpy as np
import math
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Flatten, Input, merge, Lambda, Activation, Conv2D, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
from keras.layers import merge
from keras import backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.disable_eager_execution()

class CriticNetwork(object):
    def __init__(self,sess, state_size, action_size, TAU, LEARNING_RATE):

        self.sess = sess
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE
        self.action_size = action_size
        
  

        #Now create the model
        self.model, self.action, self.state = self.create_critic_network(state_size, action_size)  
        self.target_model, self.target_action, self.target_state = self.create_critic_network(state_size, action_size)  
        self.action_grads = tf.gradients(self.model.output, self.action)  #GRADIENTS for policy update
        # self.sess.run(tf.initialize_all_variables())

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def create_critic_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape= (84, 84, 4))  
        A = Input(shape=(3, ))   

        x = Conv2D(16, kernel_size = 8, strides = (3,3), activation = 'relu')(S)
        x = Conv2D(32, kernel_size = 4, strides = (2,2), activation = 'relu')(x)
        x = Flatten()(x)
        y = Dense(1000, activation="relu")(A)
        h2 = Dense(1000, activation = "relu")(x)
        h3 = merge.Add()([h2,y])
        # h3 = Reshape((256, 1))(h3_prime)
        h4_prime = Dense(500, activation="relu")(h3)
        h4 = Dense(100, activation="relu")(h4_prime)
        V = Dense(1, activation = 'linear')(h4)
        model = Model(input=[S,A],output=V)
        adam = Adam(lr=self.LEARNING_RATE)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S 
