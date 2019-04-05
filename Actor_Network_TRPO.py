from utility import *
import tensorflow.contrib.slim as slim

action_bound = 2.0
import random


class Actor_Net():
    def __init__(self, action_dim, name, action_bound, state_dim, learning_rate=0.01, batch_size=32):
        # super().__init__(num_actions, name)
        self.learning_rate = learning_rate
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.name = name
        self._build_model()

    def _build_model(self):
        print("build " + self.name)

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)
        self.inp = tf.placeholder(shape=[None, self.state_dim[0], self.state_dim[1], self.state_dim[2]],
                                  dtype=tf.float32, name="input_image")
        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="advantages")
        self.old_mean = tf.placeholder(dtype=tf.float32, name="old_mean")
        self.old_sigma = tf.placeholder(dtype=tf.float32, name="old_sigma")

        self.conv = tf.layers.conv2d(
            inputs=self.inp,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=regularizer,
            name="conv_1")
        # self.pool = tf.layers.max_pooling2d(inputs=self.conv, pool_size=[2, 2], strides=2, name="pool_1")
        # self.norm = tf.layers.batch_normalization(self.pool, name="batch_norm_1")
        self.conv1 = tf.layers.conv2d(
            inputs=self.conv,
            filters=8,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_regularizer=regularizer, name="conv_2")

        # self.pool1 = tf.layers.max_pooling2d(inputs=self.conv1, pool_size=[2, 2], strides=2, name="pool_2")

        self.flat = tf.contrib.layers.flatten(self.conv1)

        self.dense = tf.layers.dense(self.flat, 64, activation=tf.nn.relu, name="dense_1")
        # self.dense2 = tf.layers.dense(self.flat, 64, activation=tf.nn.relu, name="dense_2")
        self.dropout = tf.layers.dropout(self.dense, rate=0.5, name="dropout_1")
        # self.dropout2 = tf.layers.dropout(self.dense2, rate=0.5,  name="dropout_2")
        self.mean = tf.scalar_mul(self.action_bound,
                                  tf.layers.dense(self.dropout, self.action_dim.shape[0], activation=tf.nn.tanh))
        # self.sigma = tf.layers.dense(self.dropout, 1, activation=tf.nn.relu)

        l2_loss = tf.losses.get_regularization_loss()

        # self.sigma = tf.clip_by_value(t=self.sigma,
        #                               clip_value_min=0,
        #                               clip_value_max=tf.sqrt(self.action_bound))
        self.sigma = tf.constant(1.)
        self.dist = tf.distributions.Normal(self.mean, self.sigma)
        self.scaled_out = self.dist.sample()
        self.net_params = tf.trainable_variables(scope=self.name)

        self.prev_mean = 0.
        self.prev_sigma = 1.
        self.prev_dist = tf.distributions.Normal(self.prev_mean, self.prev_sigma)

        # self.prev_scaled_out = self.prev_dist.sample()

        self.cost = tf.reduce_mean(
            tf.distributions.kl_divergence(self.dist, self.prev_dist, allow_nan_stats=False) * self.advantage)
        # self.actor_gradients = [tf.multiply(grad, 1/self.batch_size) for grad in tf.gradients(ys=self.output, xs=self.net_params, grad_ys=-self.action_gradients)]

        self.grads = [tf.multiply(grad, 1 / self.batch_size) for grad in
                      tf.gradients(self.cost, self.net_params, gate_gradients=False)]
        self.shapes = [v.shape.as_list() for v in self.net_params]

        self.p = tf.placeholder(tf.float32, name="p")
        tangents = []
        start = 0
        for shape in self.shapes:
            size = np.prod(shape)
            tangents.append(tf.reshape(self.p[start:start + size], shape))
            start += size
        # self.gvp =tf.map_fn( [tf.reduce_sum(g * self.p) for (g, self.p) in zip(self.grads, self.p)])
        # self.gvp = [tf.reduce_sum(tf.multiply(g ,self.p)) for (g, self.p) in zip(self.grads, self.p)]
        self.gvp = tf.add_n([(tf.reduce_sum(g * t)) for (g, t) in zip(self.grads, tangents)])

        # 2nd gradient of KL w/ itself * tangent

        self.hvp = flatgrad(self.gvp, self.net_params)

        def model_summary():
            # model_vars = tf.trainable_variables()
            slim.model_analyzer.analyze_vars(self.net_params, print_info=True)

        model_summary()
        self.saver = tf.train.Saver()

    def choose(self, sess, states, epsilon):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """

        feed = {self.inp: states}
        a = sess.run(self.output, feed)
        if random.random() < epsilon:
            return np.minimum(np.maximum(a[0] + np.random.rand(2) * 0.1 - 0.05, np.asarray([-1.0, -1.0])),
                              np.asarray([1.0, 1.0]))
        else:
            return a[0]

    def conjugate_gradient(self, f_Ax, b, cg_iters=5, residual_tol=1e-5):
        p = b.copy()
        r = b.copy()
        x = np.zeros_like(b)
        rdotr = np.dot(r, r)
        for i in range(cg_iters):
            z = f_Ax(p)
            y = np.dot(p, z) + 1e-16
            v = rdotr / y  # p.dot(z)  # stepdir size?? =ak of wikipedia
            x += v * p
            # x += v * p  # new parameters??
            r -= z * v  # new gradient??
            newrdotr = np.dot(r, r)  #
            mu = newrdotr / rdotr  # Bi of wikipedia
            rdotr = newrdotr
            p = r + mu * p

            if newrdotr < residual_tol:
                break

        return x

    def save(self, sess):
        self.saver.save(sess, "./Saved_models/model_" + self.name + ".ckpt", write_meta_graph=False)
        print(self.name + " saved")

    def load(self, sess):
        self.saver.restore(sess, "./Saved_models/model_" + self.name + ".ckpt")
        print(self.name + " loaded")

    def linesearch(self, f, x, fullstepdir, expected_improve_rate, max_iter=10):
        '''
        :param f: loss fuction
        :param x: parameters
        :param fullstepdir: value returned by conjugate gradient * Hg-1 ... delta kappa estimated by the conjugate gradient
        :param expected_improve_rate:
        :return:
        '''

        # stepsize = self.linesearch(loss, net, self.stepdir,
        #                            self.cg.dot(self.stepdir))
        j = max_iter
        accept_ratio = .1
        max_backtracks = 10
        fval = f(x)
        for (_n_backtracks, stepdirfrac) in enumerate(.5 ** np.arange(max_backtracks)):
            j -= 1
            xnew = x + (stepdirfrac * fullstepdir)
            newfval = f(xnew)
            actual_improve = fval - newfval
            expected_improve = expected_improve_rate * stepdirfrac
            ratio = actual_improve / expected_improve
            if ratio > accept_ratio and actual_improve > 0 or j == 0:
                return xnew

        return x

    def predict(self, sess, states):
        """
        Args:
          sess: TensorFlow session
          states: array of states for which we want to predict the actions.
        Returns:
          The prediction of the output tensor.
        """
        feed = {self.inp: states}
        prediction = sess.run(self.scaled_out, feed)

        return prediction

    def update(self, sess, states, advantages, summary, first):
        """
        Updates the weights of the neural network, based on its targets, its
        predictions, its loss and its optimizer.

        Args:
          sess: TensorFlow session.
          states: [current_state] or states of batch
          actions: [current_action] or actions of batch
          targets: [current_target] or targets of batch
        """

        feed_dict = {self.inp: states, self.old_mean: self.prev_mean, self.old_sigma: self.prev_sigma,
                     self.advantage: advantages, self.p: None}

        self.prev_mean, self.prev_sigma, _, cost, net, grads = sess.run(
            (self.mean, self.sigma, self.scaled_out, self.cost, self.net_params, self.grads),
            feed_dict)  # cost and gradient are fine
        # for g in grads:

        # grads = tf.where(tf.is_nan(grads), tf.zeros_like(has_nans), has_nans).eval())
        i = 0
        for g in grads:
            # print(g)
            g = np.where(np.isnan(g), 0.0, g)
            grads[i] = g
            i += 1
        net = np.concatenate([np.reshape(v, [np.prod(v.shape)]) for v in net], 0)

        def get_hvp(p):
            feed_dict[self.p] = p  # np.reshape(p, [np.size(p),1])
            a = sess.run(self.hvp, feed_dict)
            # print("ppppppppppppppp")
            # print(p)
            a = a + 1e-3 * p
            return a
            # gvp = sess.run(self.gvp, feed_dict)
            #
            # gvp = np.where(np.isnan(gvp), 1e-6, gvp)
            # print("gvp")
            # print(gvp)
            # a = tf.stop_gradient(gvp)
            # a = sess.run(a)#
            # a = tf.gradients(gvp, net, gate_gradients=True)
            # print(a)

            # print("a****************")
            # a = [np.zeros_like(p) if k is None else k for k in a]
            #
            # print(a)
            # a = np.concatenate([np.reshape(grad, [np.size(v)]) for (v, grad) in zip(net, a)], 0)

            # return 1e-3 * p + a[0]
            # return np.sum(1e-3 * np.reshape(p, [np.size(p), 1]) + np.reshape(a, [1, np.size(a)]), 1)

        grads = np.concatenate([np.reshape(grad, [np.prod(v.shape)]) for (v, grad) in zip(self.net_params, grads)], 0)

        # grads = np.where(np.isnan(grads), 1e-3, np.array(grads))
        # grads = np.array(grads[0])
        self.cg = self.conjugate_gradient(get_hvp, -grads)
        print("coniugate gradient [0]")
        print(self.cg[0])
        delta = 0.5 * self.cg * get_hvp(self.cg)
        prev_params = self.net_params

        self.stepdir = np.dot(np.sqrt(2 * self.learning_rate / (np.dot(grads, self.cg) + 1e-16)) , self.cg)
        print("stepdire [0]")
        print(self.stepdir[0])

        def loss(th):
            start = 0
            for i, shape in enumerate(self.shapes):
                size = np.prod(shape)
                self.net_params[i] = tf.reshape(th[start:start + size], shape)
                start += size
            self.prev_mean, self.prev_sigma, _, cost = sess.run(
                (self.mean, self.sigma, self.scaled_out, self.cost), feed_dict)

            return cost

        exp_improve_rate = self.cg.dot(self.stepdir)

        stepsize = self.linesearch(loss, net, self.stepdir, exp_improve_rate)
        i = 0
        start = 0
        for (shape, v) in zip(self.shapes, self.net_params):
            size = np.prod(shape)
            self.net_params[i] = prev_params[i] + tf.reshape(stepsize[start:start + size], shape)
            start += size
            i += 1


class Actor_Target_Network(Actor_Net):
    """k
    Slowly updated target network. Tau indicates the speed of adjustment. If 1,
    it is always set to the values of its associate.
    """

    def __init__(self, action_dim, name, action_bound, state_dim, actor, learning_rate=0.001,
                 tau=0.001):  # modified line
        super().__init__(action_dim, name, action_bound, state_dim, learning_rate)
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

    def get_old_mean_and_sigma(self):
        return self.old_mean, self.old_sigma
