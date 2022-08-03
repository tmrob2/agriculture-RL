import tensorflow as tf
from tensorflow import keras
import math

class NNModel(keras.Model):
    """
    Simple A2C network for computing non-image based
    data. The class inherits from keras.Model which enables 
    integration into the streamlined Keras methods of training.

    There are two dense layers with 64 nodes each. Then a Value 
    layer with one output is creted which evaluated V(s).

    Finally the policy layer output is equal to the size of the
    action space. The action output produces logits only. The
    softmax function which creates psuedo-probabilities P(s_t, a_t)
    can be applied within TensorFlow functions. 
    """
    def __init__(self, num_actions, num_tasks):
        super().__init__()
        self.num_actions = num_actions
        self.dense1 = keras.layers.Dense(
            128,
            activation='tanh'
        )
        self.dense2 = keras.layers.Dense(
            64, 
            activation='tanh'
        )
        self.value = keras.layers.Dense(1 + num_tasks)
        self.policy_logits = keras.layers.Dense(num_actions)
    
    def call(self, inputs):
        """
        Function is run whenever a state needs to be run through a 
        model, and produces a value and policy logits according 
        to the layer defintions on initialisation. 

        The Keras model API will use this function in it predict 
        functions and also its training functions. The input
        is passed through the two common Dense layers and the function
        rerturns first the value output then the policy logits
        """
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.policy_logits(x), self.value(x)
