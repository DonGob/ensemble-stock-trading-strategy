import gym
import numpy as np
from actor_critic import Agent
from  stable_baselines.common.policies import CnnLnLstmPolicy
import tensorflow as tf

def get_model():
    session = tf.compat.v1.Session()
    return CnnLnLstmPolicy(sess=session,)


