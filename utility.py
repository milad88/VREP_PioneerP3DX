import numpy as np
from collections import namedtuple
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
import itertools
from math import sqrt
EpisodeStats = namedtuple("Stats", ["episode_lengths", "episode_rewards"])
batch_size = 32
# print(np.multiply(0.5, [0.2, 0.0, 0.1]) + [1,2,2])
# # print(np.multiply([0.2, 0.0, 0.1] , 2))
# a = np.random.rand(2)
# print(a)
# print(a * 2 -1)
def Dist(x1, x2, y1, y2):
    dx = x2 - x1
    dy = y2 - y1
    Distq = dx ** 2 + dy ** 2
    rix = sqrt(Distq)
    return rix

def var_shape(x):
    try:
        out = [k for k in x.shape]
    except:
        out = [1]
    assert all(isinstance(a, int) for a in out), \
        "shape function assumes that shape is fully known"
    return out

def flatgrad(loss, var_list):
    grads = tf.gradients(loss, var_list)
    # grads = [1e-16 if g is None else g for g in grads]
    return tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)


def gauss_log_prob(mu, logstd, x):
    var = tf.exp(2 * logstd)
    gp = -tf.square(x - mu) / (2 * var) - .5 * tf.log(tf.constant(2 * np.pi)) - logstd
    return tf.reduce_sum(gp, [1])


def gauss_prob(mu, std, xs):
    var = std ** 2
    return tf.exp(-tf.square(xs - mu) / (2 * var)) / (tf.sqrt(tf.constant(2 * np.pi)) * std)

# KL divergence between two paramaterized guassian distributions
def gauss_KL(mu1, std1, mu2, std2):
    var1 = std1 ** 2
    var2 = std2 ** 2

    kl = tf.reduce_sum(tf.log(std2) - tf.log(std1) + (var1 + tf.square(mu1 - mu2)) / (2 * var2) - 0.5)

    return kl


def gauss_ent(mu, std):
    h = tf.reduce_sum(tf.log(std) + tf.constant(0.5 * np.log(2 * np.pi * np.e), tf.float32))
    return h


def hessian_vec_bk(ys, xs, vs, grads=None):
    """Implements Hessian vector product using backward on backward AD.
  Args:
    ys: Loss function.
    xs: Weights, list of tensors.
    vs: List of tensors to multiply, for each weight tensor.
  Returns:
    Hv: Hessian vector product, same size, same shape as xs.
  """
    # Validate the input

    if type(xs) == list:
        if len(vs) != len(xs):
            raise ValueError("xs and vs must have the same length.")

    if grads is None:
        grads = tf.gradients(ys, xs, gate_gradients=True)
    return tf.gradients(grads, xs, vs, gate_gradients=True)



def plot_stats(stats):
    fig11 = plt.figure(figsize=(10, 5))

    plt.plot(np.ravel(stats))
    plt.xlabel("Episode")
    plt.ylabel("loss per episode")
    plt.show(fig11)


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    #rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(stats.episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    fig2.savefig('reward.png')
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)


class SetFromFlat(object):

    def __init__(self, session, var_list):
        self.session = session
        self.var_list = var_list
        self.shapes = map(var_shape, var_list)
        total_size = sum(np.prod(shape) for shape in self.shapes)
        self.theta = theta = tf.placeholder(tf.float32, [total_size],name="theta_sff")
        start = 0
        assigns = []
        for (shape, v) in zip(self.shapes, self.var_list):
            size = np.prod(shape)
            assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
            start += size

        self.op = assigns

    def __call__(self, theta):
        self.session.run(self.op, feed_dict={self.theta: theta})


class ReplayBuffer:
    # Replay buffer for experience replay. Stores transitions.
    def __init__(self):
        self._data = namedtuple("ReplayBuffer", ["states", "actions", "next_states", "rewards", "dones"])
        self._data = self._data(states=[], actions=[], next_states=[], rewards=[], dones=[])
        self.position = 0
        self.capacity = batch_size * 4

    def add_transition(self, state, action, next_state, reward, done):

        if len(self._data.states) < self.capacity:
            self._data.states.append(None)
            self._data.actions.append(None)
            self._data.next_states.append(None)
            self._data.rewards.append(None)
            self._data.dones.append(None)

        self._data.states[self.position] = state
        self._data.actions[self.position] = action
        self._data.next_states[self.position] = next_state
        self._data.rewards[self.position] = reward
        self._data.dones[self.position] = done
        self.position = (self.position + 1) % self.capacity

    def next_batch(self, batch_size):

        self.capacity = batch_size * 4
        if batch_size >= len(self._data.states):
            return np.array(self._data.states), np.array(self._data.actions), np.array(
                self._data.next_states), np.array(self._data.rewards), np.array(self._data.dones)

        else:
            batch_indices = np.random.choice(len(self._data.states), batch_size)
            batch_states = np.array([self._data.states[i] for i in batch_indices])

            batch_actions = np.array([self._data.actions[i] for i in batch_indices])
            batch_next_states = np.array([self._data.next_states[i] for i in batch_indices])
            batch_rewards = np.array([self._data.rewards[i] for i in batch_indices])
            batch_dones = np.array([self._data.dones[i] for i in batch_indices])

            return batch_states, batch_actions, batch_next_states, batch_rewards, batch_dones

    #        if self.transition_size() > 5*batch_size:
    #           self._data.states = self._data.states[-5*batch_size:]
    #          self._data.actions = self._data.actions[-5*batch_size:]
    #         self._data.next_states = self._data.next_states[-5*batch_size:]
    #       self._data.dones = self._data.dones[-5*batch_size:]
    #      self._data.rewards = self._data.rewards[-5*batch_size:]


    def transition_size(self):
        return len(self._data.states)
