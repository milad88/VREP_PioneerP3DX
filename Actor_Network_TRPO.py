from utility import *
import itertools

action_bound = 2.0


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
                                  dtype=tf.float32)
        self.advantage = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="advantages")
        self.old_mean = tf.placeholder(dtype=tf.float32, name="old_mean")
        self.old_sigma = tf.placeholder(dtype=tf.float32, name="old_sigma")
        self.p = tf.placeholder(tf.float32, name="p")

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
        self.dense2 = tf.layers.dense(self.flat, 64, activation=tf.nn.relu)
        self.dropout = tf.layers.dropout(self.dense, rate=0.5)
        self.dropout2 = tf.layers.dropout(self.dense2, rate=0.5)
        self.mean = tf.scalar_mul(self.action_bound, tf.layers.dense(self.dropout, self.action_dim.shape[0], activation=tf.nn.tanh))
        self.sigma = tf.layers.dense(self.dropout2, 1, activation=tf.nn.relu)

        l2_loss = tf.losses.get_regularization_loss()


        self.sigma = tf.clip_by_value(t=self.sigma,
                                      clip_value_min=0,
                                      clip_value_max=tf.sqrt(self.action_bound))
        self.scaled_out = tf.truncated_normal(mean=self.mean, stddev=self.sigma, shape=[self.action_dim.shape[0]])
        self.net_params = tf.trainable_variables(scope=self.name)

        self.prev_mean = 0.
        self.prev_sigma = 1.

        self.cost = tf.reduce_sum((gauss_prob(self.mean, self.sigma, self.scaled_out) * self.advantage) /
                                  (gauss_prob(self.prev_mean, self.prev_sigma, self.scaled_out)) + 1e-10) + l2_loss

        self.grads = tf.gradients(self.cost, self.net_params)

        self.shapes = [v.shape.as_list() for v in self.net_params]

        tangents = []
        start = 0
        for shape in self.shapes:
            size = np.prod(shape)
            tangents.append(tf.reshape(self.p[start:start + size], shape))
            start += size
        # self.gvp = tf.add_n([tf.reduce_sum(g * tangent) for (g, tangent) in zip(grads, tangents)])
        self.gvp = [(tf.reduce_sum(g * t)) for (g, t) in zip(self.grads, tangents)]
        # 2nd gradient of KL w/ itself * tangent

        self.hvp = flatgrad(self.gvp, self.net_params)

        self.saver = tf.train.Saver()

    def conjugate_gradient(self, f_Ax, b, cg_iters=5, residual_tol=1e-5):
        p = b.copy()
        r = b.copy()
        x = np.zeros_like(b)
        rdotr = r.dot(r)
        for i in range(cg_iters):
            z = f_Ax(p)
            v = rdotr / p.dot(z)  # p.dot(z)  # stepdir size?? =ak of wikipedia
            x += np.dot(v, p)
            # x += v * p  # new parameters??
            r -= z.dot(v)  # new gradient??
            newrdotr = np.dot(r, r)  #
            if newrdotr < residual_tol:
                break

            mu = newrdotr / rdotr  # Bi of wikipedia
            rdotr = newrdotr
            p = r + mu * p

        return x

    def save(self, sess):
        self.saver.save(sess, "./Saved_models/model_" + self.name+".ckpt")
        print(self.name+" saved")

    def load(self, sess):
        self.saver.restore(sess, "./Saved_models/model_" + self.name+".ckpt")
        print(self.name+" loaded")


    def linesearch(self, f, x, fullstepdir, expected_improve_rate, max_iter=5):
        '''
        :param f: loss fuction
        :param x: parameters
        :param fullstepdir: value returned by conjugate gradient * Hg-1 ... delta kappa estimated by the conjugate gradient
        :param expected_improve_rate:
        :return:
        '''
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
                     self.advantage: advantages}

        self.prev_mean, self.prev_sigma, _, _, net, grads = sess.run(
            (self.mean, self.sigma, self.scaled_out, self.cost, self.net_params, self.grads), feed_dict)

        grads = np.concatenate([np.reshape(grad, [np.size(v)]) for (v, grad) in zip(net, grads)], 0)
        grads = np.where(np.isnan(grads), 1e-16, grads)

        def get_hvp(p):
            feed_dict[self.p] = p  # np.reshape(p, [np.size(p),1])
            gvp = sess.run(self.gvp, feed_dict)
            gvp = np.where(np.isnan(gvp), 0, gvp)
            # with tf.control_dependencies(self.gvp):
            a = tf.gradients(gvp, self.net_params)
            a = [0 if k is None else k for k in a]
            #            a = np.concatenate([np.reshape(grad, [np.size(v)]) for (v, grad) in zip(net, a)], 0)

            return np.sum((1e-3 * np.reshape(p, [np.size(p), 1])) + np.reshape(a, [1, np.size(a)]), 1)

            # return np.array(flatgrad(self.gvp, self.net_params))# + 1e-3 * p

        self.cg = self.conjugate_gradient(get_hvp, -grads)
        self.stepdir = np.sqrt(2 * self.learning_rate / (np.transpose(grads) * self.cg) + 1e-16) * self.cg

        def loss(th):
            start = 0
            i = 0
            for (shape, v) in zip(self.shapes, self.net_params):
                size = np.prod(shape)
                self.net_params[i] = tf.reshape(th[start:start + size], shape)
                start += size
                i += 1
            return sess.run(self.cost, feed_dict)

        stepsize = self.linesearch(loss, np.concatenate([np.reshape(g, [-1]) for g in net], 0), self.stepdir,
                                   self.cg.dot(self.stepdir))

        for i, v in enumerate(self.net_params):
            try:
                for k in range(len(v)):
                    self.net_params[i][k] -= stepsize[i][k] * self.net_params[i][k]
            except:
                self.net_params[i] -= stepsize[i] * self.net_params[i]


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

        op_holder = []
        for idx, var in enumerate(target_vars):  # // is to retun un integer
            op_holder.append(var.assign(
                (actor_vars[idx].value() * self.tau) + ((1 - self.tau) * var.value())))
        return op_holder

    def update(self, sess):
        for op in self._associate:
            sess.run(op)

    def get_old_mean_and_sigma(self):
        return self.old_mean, self.old_sigma
