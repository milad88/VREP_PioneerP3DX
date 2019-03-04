import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import numpy as np


class Critic_Net():
    def __init__(self, action_dim, name, action_bound, state_dim, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.name = name
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_dim = state_dim
        self._build_model()

    def _build_model(self):
        # self.action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim])
        print("build " + self.name)
        self.inp = tf.placeholder(shape=[None, self.state_dim[0], self.state_dim[1], self.state_dim[2]],
                                  dtype=tf.float32)

        # self.inp_act = tf.concat([self.inp, self.action], 1)
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        conv = tf.layers.conv2d(
            inputs=self.inp,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=regularizer)
        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)
        pool = tf.layers.batch_normalization(pool)
        conv = tf.layers.conv2d(
            inputs=pool,
            filters=16,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=regularizer)

        pool = tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)

        pool = tf.contrib.layers.flatten(pool)

        dense = tf.layers.dense(pool, 64, activation=tf.nn.relu)

        self.dropout = tf.layers.dropout(dense, rate=0.35)
        self.dense = tf.layers.dense(self.dropout, 1, activation=tf.nn.relu)

        self.y_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.trainer = tf.train.AdamOptimizer(self.learning_rate)

        l2_loss = tf.losses.get_regularization_loss()
        self.loss = tf.reduce_mean(tf.squared_difference(self.dense, self.y_)) + l2_loss

        self.step = self.trainer.minimize(self.loss)
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.action_grads = tf.gradients(self.dense, self.inp)

        self.saver = tf.train.Saver()

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """

        feed = {self.inp: states}  # , self.action: actions
        prediction = sess.run(self.dense, feed)

        return prediction

    def update(self, sess, states, targets, summary):  # after states actions
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """
        return sess.run((self.loss, self.dense, self.step),
                        feed_dict={self.inp: states, self.y_: targets})  # self.action: actions,

    def action_gradients(self, sess, states):  # actions as well
        return sess.run(self.action_grads, feed_dict={
            self.inp: states})  # self.action: actions

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

    def __init__(self, action_dim, name, action_bound, state_dim, learning_rate=0.001, tau=0.001):
        super().__init__(action_dim, name, action_bound, state_dim, learning_rate)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):

        critic_vars = tf.trainable_variables()  # "critic"
        target_vars = tf.trainable_variables()  # "critic_target"

        op_holder = []
        for idx, var in enumerate(target_vars):  # // is to retun un integer
            op_holder.append(var.assign(
                (critic_vars[idx].value() * self.tau) + ((1 - self.tau) * var.value())))
        # return target_vars.assign((critic_vars * self.tau )+((1 - self.tau) * target_vars))
        return op_holder

    def update(self, sess):
        for op in self._associate:
            sess.run(op)
