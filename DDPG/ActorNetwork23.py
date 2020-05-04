import numpy as np
import math
from keras.models import model_from_json
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Input, merge, Lambda, Conv2D, Concatenate, concatenate
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.compat.v1.disable_eager_execution()

class ActorNetwork(object):
    def __init__(self,sess, state_size, action_size, TAU, LEARNING_RATE):
   
        self.sess = sess
        self.TAU = TAU
        self.LEARNING_RATE = LEARNING_RATE

  

        #Now create the model
        self.model , self.weights, self.state = self.create_actor_network(state_size, action_size)   
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(state_size, action_size) 
        self.action_gradient = tf.compat.v1.placeholder(tf.float32,[None, action_size])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.optimizers.Adam(LEARNING_RATE).apply_gradients(grads)
        # self.sess.run(tf.initialize_all_variables())

    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1 - self.TAU)* actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)

    def create_actor_network(self, state_size,action_dim):
        print("Now we build the model")
        S = Input(shape= (84, 84, 4))    

        x = Conv2D(16, kernel_size = 8, strides = (3,3), activation = 'relu')(S)
        x = Conv2D(32, kernel_size = 4, strides = (2,2), activation = 'relu')(x)
        x = Flatten()(x)
        x = Dense(500, activation="relu")(x)
        x = Dense(100, activation = "relu")(x)
        steering = Dense(1,activation='tanh')(x)
        acceleration = Dense(1,activation='sigmoid')(x)
        brake = Dense(1,activation='sigmoid')(x)

        V =  concatenate([steering,acceleration,brake])  

        model = Model(input=S,output=V)
        return model, model.trainable_weights, S

