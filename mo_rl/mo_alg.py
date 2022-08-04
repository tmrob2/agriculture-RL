import keras.losses
import numpy as np
from typing import List, Tuple
import gym
import tensorflow as tf
from mo_rl.dfa import DFA
import copy

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
ENTROPY_COEF = 1

class EnvMem:
    def __init__(self, envs: List[gym.Env], tasks,):
        self.envs = envs
        self.dfas = {k: copy.deepcopy(tasks) for k in range(len(envs))}
        self.num_tasks = len(tasks)
        self.data = {k: [] for k in range(self.num_tasks)}

    def env_step(self, action: np.int, agent: np.int, data: np.int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        state, reward, done, info = self.envs[agent].step(action)
        if np.int_(data) == 1:
            self.data[np.int_(agent)].append(info['crops_planted'])
        # plug the info into the dfa.next function
        task_rewards = [0.] * self.num_tasks
        task_states = [-1] * self.num_tasks
        _reward_dict = info['reward']
        date = None
        for k, v in _reward_dict.items():
            date = k
        for (j, task) in enumerate(self.dfas[np.int_(agent)]):
            task.current_state, task_reward = task.next(task.current_state, info['crops_planted'], date)
            task_states[j] = task.current_state
            task_rewards[j] = task_reward
        product_done = False
        if all(task.check_done for task in self.dfas[np.int_(agent)]) or done:
            product_done = True
        product_state = np.concatenate([task_states, state], dtype=np.float32)
        product_rewards = np.concatenate([[reward], task_rewards], dtype=np.float32)
        return product_state.astype(np.float32), np.array(product_rewards, np.float32), np.array(product_done, np.int32)

    def sample_action(self, env_id):
        return self.envs[env_id].action_space.sample()

    def tf_env_step(self, action: tf.Tensor, agent: tf.Tensor, data: tf.Tensor) -> List[tf.Tensor]:
        return tf.numpy_function(self.env_step, [action, agent, data], [tf.float32, tf.float32, tf.int32])

    def reset(self, agent):
        env_obs = self.envs[agent].reset()
        return self._process_obs(env_obs, agent)

    def tf_reset(self, env_id):
        return tf.numpy_function(self.reset, [env_id], tf.float32)

    def _process_obs(self, env_obs, agent):
        [dfa.reset() for dfa in self.dfas[np.int_(agent)]]
        q = [dfa.current_state for dfa in self.dfas[np.int_(agent)]]
        full_state = np.concatenate([q, env_obs])
        return full_state

    def task_rewards(self):
        return np.array([1.0 if dfa.progress['jf'] else 0 for dfa in self.dfas])

    def tf_task_rewards(self):
        return tf.numpy_function(self.task_rewards, [], tf.float32)

    def run_episode(
        self,
        initial_state: tf.Tensor,
        model: tf.keras.Model,
        max_steps: int,
        agent: tf.int32,
        info: tf.int32
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Runs a single episode to collect training data"""
        action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

        initial_state_shape = initial_state.shape
        state = initial_state

        for t in tf.range(max_steps):
            # Convert state into a batched tensor
            state1 = tf.expand_dims(state, 0)

            # Run the model to get action probabilties and a critic value
            action_logits_t, value = model(state1)

            # Sample the next action from the action probability distribution
            action = tf.random.categorical(action_logits_t, 1)[0, 0]
            action_probs_t = tf.nn.softmax(action_logits_t)

            # Store the critic values
            values = values.write(t, tf.squeeze(value))

            # Store log probability of the action chose
            action_probs = action_probs.write(t, action_probs_t[0, action])

            # Apply the action to the environment to get the next state and reward
            state, reward, done = self.tf_env_step(action, agent, info)
            state.set_shape(initial_state_shape)

            # Store rewards
            rewards = rewards.write(t, reward)

            if tf.cast(done, tf.bool):
                # print(f"agent {agent} ran for {t} step")
                break
        action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()

        self.tf_reset(agent)
        return action_probs, values, rewards

    def get_expected_returns(self, rewards: tf.Tensor, gamma: float) -> tf.Tensor:
        """Compute the expected rewards per timestep"""
        n = tf.shape(rewards)[0]
        returns = tf.TensorArray(dtype=tf.float32, size=n)
        # Start form the end of the rewards and accumulate reward sums
        # into the returns array
        rewards = tf.cast(rewards[::-1], dtype=tf.float32)
        discounted_sum = tf.constant([0.] * (self.num_tasks + 1))
        discounted_sum_shape = discounted_sum.shape
        for i in tf.range(n):
            reward = rewards[i]
            discounted_sum = reward + gamma * discounted_sum
            discounted_sum.set_shape(discounted_sum_shape)
            returns = returns.write(i, discounted_sum)
        returns = returns.stack()[::-1]
        return returns


def df(x, c) -> tf.Tensor:
    if x <= c:
        return 2 * (x - c)
    else:
        return tf.convert_to_tensor(0.0)

def dh(x, e) -> tf.Tensor:
    if x <= e:
        return 2 * (x - e)
    else:
        return tf.convert_to_tensor(0.0)

def computeH(X: tf.Tensor, Xi: tf.Tensor, lam, chi, mu, e, c) -> tf.Tensor:
    _, y = X.get_shape()
    H = [lam * df(Xi[0], c)]
    for j in range(1, y):
        H.append(chi * dh(tf.math.reduce_sum(mu * X[:, j]), e[j - 1]) * mu)
    return tf.expand_dims(tf.convert_to_tensor(H), 1)

def compute_loss(
    action_probs: tf.Tensor,
    values: tf.Tensor,
    returns: tf.Tensor,
    ini_value: tf.Tensor,
    ini_values_i: tf.Tensor,
    lam,
    chi,
    mu,
    e,
    c
) -> tf.Tensor:
    """Computes the combined actor-critic loss."""
    print("init value ", ini_values_i)
    H = computeH(ini_value, ini_values_i, lam, chi, mu, e, c)
    advantage = tf.matmul(returns - values, H)
    action_log_probs = tf.math.log(action_probs)
    actor_loss = tf.math.reduce_sum(action_log_probs * advantage)
    critic_loss = huber_loss(values, returns)
    print(f"actor loss: {actor_loss}, critic loss: {critic_loss * 1000}, loss: {actor_loss + critic_loss * 1000}")
    entropy = ENTROPY_COEF * keras.losses.categorical_crossentropy(tf.squeeze(action_probs), tf.squeeze(action_probs))
    return actor_loss + 1000 * critic_loss - entropy


