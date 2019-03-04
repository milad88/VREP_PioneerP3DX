import numpy as np
import tensorflow as tf

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

        print("build "+ self.name)

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        self.inp = tf.placeholder(shape=[None, self.state_dim[0], self.state_dim[1], self.state_dim[2]],
                                  dtype=tf.float32)

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

        self.flat = tf.contrib.layers.flatten(self.pool1)

        self.dense = tf.layers.dense(self.flat, 64, activation=tf.nn.relu)
        self.dropout = tf.layers.dropout(self.dense, rate=0.5)
        self.output = tf.layers.dense(self.dropout, self.action_dim.shape[0], activation=tf.nn.tanh)

        self.scaled_outputs = tf.scalar_mul(action_bound, self.output)

        l2_loss = tf.losses.get_regularization_loss()
        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        self.action_gradients = tf.placeholder(tf.float32)
        self.actor_gradients = tf.gradients(ys=self.output, xs=self.net_params, grad_ys=-self.action_gradients)

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.actor_gradients, self.net_params))

        self.saver = tf.train.Saver()

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """

        feed = {self.inp: states}
        prediction = sess.run(self.scaled_outputs, feed)
        return prediction

    # action gradient to be fed

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

        sess.run(self.optimize, feed_dict={self.inp: states, self.action_gradients: grads[0]})

    def save(self, sess):
        self.saver.save(sess, "./Saved_models/model_" + self.name+".ckpt")
        print(self.name+" saved")

    def load(self, sess):
        self.saver.restore(sess, "./Saved_models/model_" + self.name+".ckpt")
        print(self.name+" loaded")


class Actor_Target_Network(Actor_Net):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, action_dim, name, action_bound, state_dim, learning_rate=0.001, tau=0.001):
        super().__init__(action_dim, name, action_bound, state_dim, learning_rate)
        # self._build_model( num_actions, action_dim, name, action_bound, state_dim)
        self.tau = tau
        self._associate = self._register_associate()

    def _register_associate(self):

        actor_vars = tf.trainable_variables()  # "actor"
        target_vars = tf.trainable_variables()  # "actor_target"

        # print(tf_vars)

        # total_vars = len(tf_vars)

        op_holder = []
        for idx, var in enumerate(target_vars):  # // is to retun un integer
            op_holder.append(var.assign(
                (actor_vars[idx].value() * self.tau) + ((1 - self.tau) * var.value())))
        return op_holder

    def update(self, sess):
        for op in self._associate:
            sess.run(op)
