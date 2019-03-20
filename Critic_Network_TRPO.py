import tensorflow as tf
import numpy as np

ac_dim = 1


class Critic_Net():
    def __init__(self, action_dim, name, action_bound, state_dim, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.name = name
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_dim = state_dim
        self._build_model()

    def _build_model(self):
        print("build " + self.name)
        self.inp = tf.placeholder(shape=[None, self.state_dim[0], self.state_dim[1], self.state_dim[2]],
                                  dtype=tf.float32)
        self.action = tf.placeholder(dtype=tf.float32, shape=[None, self.action_dim.shape[0]])

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

        dense = tf.layers.dense(pool, 62, activation=tf.nn.relu)

        dense = tf.concat([dense, self.action], 1)

        self.dropout = tf.layers.dropout(dense, rate=0.65)
        self.outputs = tf.layers.dense(self.dropout, 1)

        self.y_ = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.trainer = tf.train.AdamOptimizer(self.learning_rate)

        l2_loss = tf.losses.get_regularization_loss()
        self.loss = tf.losses.mean_squared_error(self.outputs, self.y_) + l2_loss

        self.net_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        self.trainer = tf.train.AdamOptimizer(self.learning_rate)
        self.loss = tf.losses.mean_squared_error(self.outputs, self.y_)

        self.step = self.trainer.minimize(self.loss)

        self.saver = tf.train.Saver()

    def predict(self, sess, states, actions):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """

        feed = {self.inp: states , self.action: actions}
        prediction = sess.run(self.outputs, feed)

        return prediction

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
        return sess.run((self.loss, self.outputs, self.step),
                        feed_dict={self.inp: states, self.action: actions, self.y_: targets})[0]  # self.action: actions,

    def save(self, sess):
        self.saver.save(sess, "./Saved_models/model_" + self.name + ".ckpt")
        print(self.name + " saved")

    def load(self, sess):
        self.saver.restore(sess, "./Saved_models/model_" + self.name + ".ckpt")
        print(self.name + " loaded")


class Critic_Target_Network(Critic_Net):
    """
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, action_dim, name, action_bound, state_dim, critic, learning_rate=0.001,
                 tau=0.001):  # modified line
        super().__init__(action_dim, name, action_bound, state_dim, learning_rate)
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
