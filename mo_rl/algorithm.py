import tensorflow as tf
from tensorflow import keras
import numpy as np

CRITIC_LOSS_WEIGHT = 0.5
ACTOR_LOSS_WEIGHT = 1.0
ENTROPY_LOSS_WEIGHT = 0.05
BATCH_SIZE = 64
GAMMA = 1.0

def critic_loss(discounted_rewards, predicted_values):
    return keras.losses.mean_squared_error(discounted_rewards, predicted_values) * \
        CRITIC_LOSS_WEIGHT

def actor_loss(combined, policy_logits):
    actions = combined[:, 0] # Array with two columns (and BATCH_SIZE rows)
    # The first column corresponds to the recorded actions of the agent as it traverses the 
    # environment. The second column corresponds to the calculated advantages
    advantages = combined[:, 1]
    # sparse categorical cross-entropy function specifies that the input of the function is
    # logits and iot also specifies the reduction to apply to the BATCH_SIZE number of 
    # calculated losses. In this case a sum function, this is the summation in
    #           J(\theta) ~ (\big_sum_{t=0}^{T-1}logP_{\pi_theta}(a_t | s_t)) A(s_t, a_t)
    sparse_ce = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction=tf.keras.losses.Reduction.SUM
    )
    # The actions are cast into tensorflow big integers
    actions = tf.cast(actions, tf.int32)
    # The policy loss is calculated based on the categorical cross-entory defined above
    # Selects those policy probabilities that correspond to the action actually taken
    # in the environment, and weights them by the advantage values
    # By allpying sum reduction the loss formula will be implemented for this function
    policy_loss = sparse_ce(actions, policy_logits, sample_weight=advantages)
    # the actual probabilities for the action are estimated by applying the softmax function 
    # to the logits
    probs = tf.nn.softmax(policy_logits)
    # The entropy loss is computed be applying the categorical cross-entropy function
    entropy_loss = keras.losses.categorical_crossentropy(probs, probs)
    return policy_loss * ACTOR_LOSS_WEIGHT - \
        entropy_loss * ENTROPY_LOSS_WEIGHT

def advantages(rewards, dones, values, next_value):
    """
    rewards: list of all the rewards that were accumulated during the agent's traversal of the 
    game.
    dones: list of 1 or 0 representing whether the env has finished an episode at each timestep
    values: is a list of all values V(s) generated by the model at each timestep
    next_value: is the bootstrapped estimatew of the value of all discounted rewards downstream
    of the last state recorded in the list
    """
    # create a numpy array of the list of rweards, with the bootstrapped next_value appended to it
    discounted_rewards = np.array(rewards + [next_value[0]])
    # proceeding backwards
    for t in reversed(range(len(rewards))):
        # compute the dicounted rewards and mask by whether the episode is done or not
        discounted_rewards[t] = rewards[t] + GAMMA * discounted_rewards[t + 1] * (1- dones[t])
    discounted_rewards = discounted_rewards[:-1]
    # compute the advantages
    advantages = discounted_rewards - np.stack(values)[:, 0]
    return discounted_rewards, advantages




