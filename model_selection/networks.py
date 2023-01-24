from curses import KEY_RESTART
import os
import tensorflow as tf
from keras.layers import Dense
import keras

class DNN(keras.model):
    def __init__(self, output_dims, activation_type=None, fc1_dims=1024, fc2_dims=512,
            name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(DNN, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_ac")
        self.output_dims = output_dims
        self.activation_type = activation_type

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.output = Dense(self.output_dims, activation=self.activation_type)

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        output = self.output(value)

        return output

class ActorCriticNetwork(keras.model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
            name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_ac")
        self.n_actions = n_actions

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi


