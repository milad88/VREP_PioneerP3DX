import numpy as np
import tensorflow as tf
import random
from tensorflow.contrib.layers import fully_connected

action_bound = 2.0


class Actor_Net():
    def __init__(self, action_dim, name, action_bound, obs_space, learning_rate=0.01, batch_size=128):
        # super().__init__(num_actions, name)
        self.learning_rate = learning_rate
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_dim = obs_space
        self.batch_size = batch_size
        self.name = name
        self._build_model()

    def _build_model(self):
        # self.action = tf.placeholder(dtype=tf.float128, shape=[None, self.action_dim])

        print("build " + self.name)

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        self.inp = tf.placeholder(shape=[None, self.state_dim[0], self.state_dim[1], self.state_dim[2]],
                                  dtype=tf.float32)
        # self.inp = tf.placeholder(shape=self.state_dim, dtype=tf.float32)
        self.conv = tf.layers.conv2d(
            inputs=self.inp,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=regularizer)
        self.pool = tf.layers.max_pooling2d(inputs=self.conv, pool_size=[2, 2], strides=2)
        self.norm = tf.layers.batch_normalization(self.pool)
        self.conv1 = tf.layers.conv2d(
            inputs=self.norm,
            filters=16,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=regularizer)

        self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2)
        self.norm1 = tf.layers.batch_normalization(self.pool1)

        self.flat = tf.contrib.layers.flatten(self.pool1)

        self.dense = tf.layers.dense(self.flat, 64, activation=tf.nn.relu)

        self.dropout = tf.layers.dropout(self.dense, rate=0.65)
        self.output = tf.layers.dense(self.dropout, self.action_dim.shape[0], activation=tf.nn.tanh)

        self.scaled_outputs = self.output
        # self.scaled_outputs = tf.scalar_mul(action_bound, self.output)

        # l2_loss = tf.losses.get_regularization_loss()
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        self.action_gradients = tf.placeholder(tf.float32)
        # self.actor_gradients = tf.gradients(ys=self.output, xs=self.net_params, grad_ys=-self.action_gradients)
        # inv_batch_size = 1/self.batch_size
        # inv_batch_size = tf.constant(1/self.batch_size, dtype=tf.float32)
        self.actor_gradients = tf.gradients(ys=self.output, xs=self.net_params, grad_ys=-self.action_gradients)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate/ self.batch_size).apply_gradients(
            zip(self.actor_gradients, self.net_params))

        self.saver = tf.train.Saver()

    def choose(self, sess, states, epsilon = 0.15):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """

        feed = {self.inp: states}
        a = sess.run(self.output, feed)
        for i in range(len(a)):
            if random.random() < epsilon:
                # a[i] = np.minimum(np.maximum(a[i] + np.random.rand(2), np.asarray([-1.0, -1.0])),
                #                   np.asarray([1.0, 1.0]))

                a[i] = np.random.rand(2) * 2 - 1
                # a[i] = np.maximum(a[i] + np.random.rand(2), np.asarray([-1.0, -1.0]))

        return a

    def predict(self, sess, state):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """

        feed = {self.inp: state}
        prediction = sess.run(self.scaled_outputs, feed)
        return prediction

    def predict_batch(self, sess, states):
        preds = []
        for s in states:
            preds.append(self.predict(sess, s))
        return preds

    def get_gradient(self, sess, s, grads):
        return sess.run(self.actor_gradients, {self.inp: s, self.action_gradients: grads})

    def update(self, sess, states, grads, summary):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """

        sess.run(self.optimize,
                              feed_dict={self.inp: states, self.action_gradients: grads})



    def update_batch(self, sess, states, grads, summay):
        batch_grads = []
        for s in states:
            batch_grads.append(self.get_gradient(sess, s, grads))

        batch_grads = tf.concat(axis=0, values=batch_grads)

        batch_grads = tf.reduce_mean(batch_grads, 0)
        sess.run(self.optimize, feed_dict={self.action_gradients: batch_grads})

    def save(self, sess):
        self.saver.save(sess, "./Saved_models/model_" + self.name + ".ckpt")
        print(self.name + " saved")

    def load(self, sess):
        self.saver.restore(sess, "./Saved_models/model_" + self.name + ".ckpt")
        print(self.name + " loaded")


class Actor_Target_Network(Actor_Net):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, action_dim, name, action_bound, state_dim, actor, learning_rate=0.001, batch_size=128,
                 tau=0.001):  # modified line
        super().__init__(action_dim, name, action_bound, state_dim, learning_rate, batch_size)
        # self._build_model( num_actions, action_dim, name, action_bound, state_dim)
        self.tau = tau
        self.actor = actor  # added line
        self._register_associate()  # modified line

    # modified method
    def _register_associate(self):
        self.init_target = [self.net_params[i].assign(self.actor.net_params[i]) for i in range(len(self.net_params))]

        self.update_target = [self.net_params[i].assign(
            tf.scalar_mul(self.tau, self.actor.net_params[i]) + tf.scalar_mul(1. - self.tau, self.net_params[i])) for i
            in range(len(self.net_params))]

    # added method. Target network starts identical to original network
    def init(self, sess):
        sess.run(self.init_target)

    # modified method
    def update(self, sess):
        sess.run(self.update_target)
