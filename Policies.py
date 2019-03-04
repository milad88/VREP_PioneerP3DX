"""
Policies
"""
import numpy as np


def make_bayesian_policy(estimator, num_actions):
    """

    :param estimator:
    :return:
    """

    def policy_fn(sess, observation, keep):
        props = np.zeros(num_actions, dtype=float)
        q_values = estimator.predict(sess, observation, keep)
        props[np.argmax(q_values)] = 1
        action = np.random.choice(np.arange(len(props)), p=props)
        return action

    return policy_fn


def make_greedy_policy(estimator, epsilon, num_actions, time_step, nTimes_actions, decay):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        num_actions: Number of actions in the environment.

    Returns:
        props function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length num_actions.

    """

    def policy_fn(sess, observation):
        props = np.zeros(num_actions, dtype=float)
        q_values = estimator.predict(sess, observation)
        props[np.argmax(q_values)] = 1
        action = np.random.choice(np.arange(len(props)), p=props)
        return action

    return policy_fn


def make_epsilon_greedy_policy(estimator, epsilon, num_actions, time_step, nTimes_actions, decay):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        num_actions: Number of actions in the environment.

    Returns:
        props function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length num_actions.

    """

    def policy_fn(sess, observation):
        props = np.ones(num_actions, dtype=float) * epsilon / num_actions
        q_values = estimator.predict(sess, observation)
        best_action = np.argmax(q_values)
        props[best_action] += (1.0 - epsilon)
        action = np.random.choice(np.arange(len(props)), p=props)
        return action

    return policy_fn


def make_epsilon_greedy_decay_policy(estimator, epsilon, num_actions, time_step, nTimes_actions, decay):
    """
    Creates an epsilon-greedy policy based on a given Q-function approximator and epsilon.

    Args:
        estimator: An estimator that returns q values for a given state
        epsilon: The probability to select a random action . float between 0 and 1.
        num_actions: Number of actions in the environment.

    Returns:
        props function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length num_actions.

    """
    eps = decay

    def policy_fn(sess, observation):
        props = np.ones(num_actions, dtype=float) * eps / num_actions
        q_values = estimator.predict(sess, observation)
        best_action = np.argmax(q_values)
        props[best_action] += (1.0 - eps)
        action = np.random.choice(np.arange(len(props)), p=props)
        return action

    return policy_fn


def make_ucb_policy(estimator, epsilon, num_actions, time_step, nTimes_actions, decay):
    # p = t ** (-4) -- probability p that true value exceeds UCB
    def policy_fn(sess, observation):
        props = np.zeros(num_actions, dtype=float)
        # observation = list(observation)
        # print(observation.shape())
        # observation = np.reshape(observation,[1,3])
        q_values = estimator.predict(sess, observation)
        for i in range(num_actions):
            val2 = np.log(time_step + 1) / nTimes_actions[i]
            val = np.sqrt(2 * val2) + q_values[0][i]
            props[i] = val
        action = np.argmax(props)
        return action

    return policy_fn
