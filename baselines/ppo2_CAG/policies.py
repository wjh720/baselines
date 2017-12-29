import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
import baselines.common.tf_util as U
import time

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=512, reuse=False):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape) #obs
        print(ob_shape)
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            with tf.variable_scope("pi"):
                h = conv(X, 'c1', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
                h2 = conv(h, 'c2', nf=128, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
                h3 = conv(h2, 'c3', nf=256, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
                h3 = conv_to_fc(h3)
                h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
                xs = batch_to_seq(h4, nenv, nsteps)
                ms = batch_to_seq(M, nenv, nsteps)
                h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
                h5 = seq_to_batch(h5)
                pi = fc(h5, 'pi', nact, act=lambda x:x)

            with tf.variable_scope("vf"):
                h = conv(X, 'c1', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
                h2 = conv(h, 'c2', nf=128, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
                h3 = conv(h2, 'c3', nf=256, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
                h3 = conv_to_fc(h3)
                h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
                xs = batch_to_seq(h4, nenv, nsteps)
                ms = batch_to_seq(M, nenv, nsteps)
                h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
                h5 = seq_to_batch(h5)
                vf = fc(h5, 'v', 1, act=lambda x:x)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps

        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = conv(tf.cast(X, tf.float32)/255., 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2))
            h2 = conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2))
            h3 = conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2))
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))
            xs = batch_to_seq(h4, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact, act=lambda x:x)
            vf = fc(h5, 'v', 1, act=lambda x:x)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.float32, ob_shape) #obs
        print(ob_shape)

        self.pdtype = pdtype = make_pdtype(ac_space)
        with tf.variable_scope("model", reuse=reuse):
            '''
            h = conv(X, 'c1', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
            h2 = conv(h, 'c2', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
            h3 = conv(h2, 'c3', nf=128, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
            h3 = conv_to_fc(h3)
            h4 = fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2))

            hh = conv(X, 'xc1', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
            hh2 = conv(hh, 'xc2', nf=64, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
            hh3 = conv(hh2, 'xc3', nf=128, rf=3, stride=1, init_scale=np.sqrt(2), pad="SAME")
            hh3 = conv_to_fc(hh3)
            hh4 = fc(hh3, 'xfc1', nh=512, init_scale=np.sqrt(2))
            pi = fc(h4, 'pi', nact, act=lambda x:x, init_scale=0.01)
            vf = fc(hh4, 'v', 1, act=lambda x:x)[:,0]

            '''
            x = tf.nn.relu(U.conv2d(X, 32, "l1", [3, 3], [1, 1], pad="SAME"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [3, 3], [1, 1], pad="SAME"))
            x = tf.nn.relu(U.conv2d(x, 128, "l3", [3, 3], [1, 1], pad="SAME"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 512, 'lin', U.normc_initializer(1.0)))

            y = tf.nn.relu(U.conv2d(X, 32, "yl1", [3, 3], [1, 1], pad="SAME"))
            y = tf.nn.relu(U.conv2d(y, 64, "yl2", [3, 3], [1, 1], pad="SAME"))
            y = tf.nn.relu(U.conv2d(y, 128, "yl3", [3, 3], [1, 1], pad="SAME"))
            y = U.flattenallbut0(y)
            y = tf.nn.relu(U.dense(y, 512, 'ylin', U.normc_initializer(1.0)))

            pi = U.dense(x, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
            vf = U.dense(y, 1, "value", U.normc_initializer(1.0))[:, 0]


        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            h1 = fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            pi = fc(h2, 'pi', actdim, act=lambda x:x, init_scale=0.01)
            h1 = fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            h2 = fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2), act=tf.tanh)
            vf = fc(h2, 'vf', 1, act=lambda x:x)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim], 
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value