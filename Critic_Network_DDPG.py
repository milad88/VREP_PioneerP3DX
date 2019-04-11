import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np
import random


class Critic_Net():
    def __init__(self, action_dim, name, action_bound, state_dim, learning_rate=0.01, batch_size=128):
        self.learning_rate = learning_rate
        self.name = name
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.state_dim = state_dim
        self._build_model()

    def _build_model(self):
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim.shape[0]])
        print("build " + self.name)
        self.inp = tf.placeholder(shape=[None, self.state_dim[0], self.state_dim[1], self.state_dim[2]],
                                  dtype=tf.float32)

        # self.inp_act = tf.concat([self.inp, self.action], 1)
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        conv = tf.layers.conv2d(
            inputs=self.inp,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=regularizer)
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
        pool = tf.layers.batch_normalization(pool)
        conv = tf.layers.conv2d(
            inputs=pool,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=regularizer)

        pool1 = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
        norm1 = tf.layers.batch_normalization(pool1)

        pool = tf.contrib.layers.flatten(norm1)

        dense = tf.layers.dense(pool, 30, activation=tf.nn.tanh)
        dense = tf.concat([dense, self.action], 1)

        self.dropout = tf.layers.dropout(dense, rate=0.5)
        self.dense = tf.layers.dense(self.dropout, 1)  # no activation

        self.y_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.trainer = tf.train.GradientDescentOptimizer(self.learning_rate)

        l2_loss = tf.losses.get_regularization_loss()
        self.loss = tf.losses.mean_squared_error(self.dense, self.y_) + l2_loss

        self.step = self.trainer.minimize(self.loss)
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.action_grads = tf.gradients(self.dense, self.action, gate_gradients=True)

        self.saver = tf.train.Saver()

    def predict(self, sess, states, actions):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """

        feed = {self.inp: states, self.action: actions}
        prediction = sess.run(self.dense, feed)
        return prediction

    def predict_batch(self, sess, states, actions):
        preds = []
        for s, a in zip(states, actions):
            preds.append(self.predict(sess, [s], [a]))
        return preds

    def update(self, sess, states, actions, targets, summary):  # after states actions
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """
        d, l, s, g = sess.run((self.dense, self.loss, self.step, self.action_grads),
                              feed_dict={self.inp: states, self.y_: targets, self.action: actions})
        print("this print is in updatefunction of crirtic")
        print(l)
        return d, l, s, g[0] / len(g[0])

    def update_batch(self, sess, states, actions, targets, summary):
        losses = []
        for s, a, t in zip(states, actions, targets):
            losses.append(self.update(sess, [s], [a], t, summary))

        print("loseesss")
        print(losses)
        # losses = np.mean(np.concatenate(losses))
        losses = np.reshape(losses, [len(losses), 1])
        sess.run(self.step, feed_dict={self.loss: losses})

    def get_critic_gradients(self, sess, states, actions):  # actions as well
        grads = []
        for s, a in zip(states, actions):
            grads.append(sess.run(self.action_grads, feed_dict={
                self.inp: states, self.action: actions}))  #

        grads = tf.concat(axis=0, values=grads)
        return tf.reduce_mean(grads, 0)

    def save(self, sess):
        save_path = self.saver.save(sess, "./Saved_models/model_" + self.name + ".ckpt")
        print(self.name + " saved")

    def load(self, sess):
        self.saver.restore(sess, "./Saved_models/model_" + self.name + ".ckpt")
        print(self.name + " loaded")


class Critic_Target_Network(Critic_Net):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, action_dim, name, action_bound, state_dim, critic, learning_rate=0.001, batch_size=128,
                 tau=0.001):  # modified line
        super().__init__(action_dim, name, action_bound, state_dim, learning_rate, batch_size)
        self.tau = tau
        self.critic = critic  # added line
        self._register_associate()  # modified line

    # modified method
    def _register_associate(self):
        self.init_target = [self.net_params[i].assign(self.critic.net_params[i]) for i in range(len(self.net_params))]

        self.update_target = [self.net_params[i].assign(
            tf.scalar_mul(self.tau, self.critic.net_params[i]) + tf.scalar_mul(1. - self.tau, self.net_params[i])) for i
            in range(len(self.net_params))]

    # added method. Target network starts identical to original network
    def init(self, sess):
        sess.run(self.init_target)

    # modified method
    def update(self, sess):
        sess.run(self.update_target)
