import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import time
from capsLayer import CapsLayer
import numpy as np

from config import cfg


class Capsule_policy(object):
    recurrent = False
    def __init__(self, name, ob_space, ac_space, kind='large'):
        with tf.variable_scope(name):
            self._init(ob_space, ac_space, kind)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, kind):
        assert isinstance(ob_space, gym.spaces.Box)


        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = cfg.batch_size

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        self.X = ob / 255.

        epsilon = 1e-8

        with tf.variable_scope('actor'):
            with tf.variable_scope('Conv1_layer'):
                # Conv1, [batch_size, 20, 20, 256]
                conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                                 kernel_size=9, stride=1,
                                                 padding='VALID')
                assert conv1.get_shape() == [cfg.batch_size, 20, 20, 256]

            # Primary Capsules layer, return [batch_size, 1152, 8, 1]
            with tf.variable_scope('PrimaryCaps_layer'):
                primaryCaps = CapsLayer(num_outputs=8, vec_len=8, with_routing=False, layer_type='CONV')
                caps1 = primaryCaps(conv1, kernel_size=9, stride=2)

            # DigitCaps layer, return [batch_size, 10, 16, 1]
            with tf.variable_scope('DigitCaps_layer'):
                digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
                self.caps2 = digitCaps(caps1)

            # Decoder structure in Fig. 2
            # 1. Do masking, how:
            with tf.variable_scope('Masking'):
                # a). calc ||v_c||, then do softmax(||v_c||)
                # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                                                      axis=2, keep_dims=True) + epsilon)
                self.softmax_v = tf.nn.softmax(self.v_length, dim=1)

                # b). pick out the index of max softmax val of the 10 caps
                # [batch_size, 10, 1, 1] => [batch_size] (index)
                self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
                self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size,))

                self.one_hot = tf.one_hot(self.argmax_idx, 10)

                # Method 1.
                if not cfg.mask_with_y:
                    # c). indexing
                    # It's not easy to understand the indexing process with argmax_idx
                    # as we are 3-dim animal

                    masked_v = []
                    for batch_size in range(cfg.batch_size):
                        v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
                        masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                    self.masked_v = tf.concat(masked_v, axis=0)
                # Method 2. masking with true label, default mode
                else:
                    # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
                    self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)))
                    self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + epsilon)

            # 2. Reconstructe the MNIST images with 3 FC layers
            # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
            with tf.variable_scope('Decoder'):
                vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
                fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=256)
                fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=512)
                self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)


        x = self.one_hot
        x = tf.nn.relu(U.dense(x, 512, 'lin', U.normc_initializer(1.0)))

        y = self.one_hot
        y = tf.nn.relu(U.dense(y, 512, 'ylin', U.normc_initializer(1.0)))

        '''
        x = ob
        if kind == 'small': # from A3C paper
            x = tf.nn.relu(U.conv2d(x, 16, "l1", [3, 3], [1, 1], pad="SAME"))
            x = tf.nn.relu(U.conv2d(x, 32, "l2", [3, 3], [1, 1], pad="SAME"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 256, 'lin', U.normc_initializer(1.0)))
        elif kind == 'large': # Nature DQN
            x = tf.nn.relu(U.conv2d(x, 64, "l1", [3, 3], [1, 1], pad="SAME"))
            x = tf.nn.relu(U.conv2d(x, 64, "l2", [3, 3], [1, 1], pad="SAME"))
            x = tf.nn.relu(U.conv2d(x, 128, "l3", [3, 3], [1, 1], pad="SAME"))
            x = U.flattenallbut0(x)
            x = tf.nn.relu(U.dense(x, 512, 'lin', U.normc_initializer(1.0)))
        else:
            raise NotImplementedError

        y = ob
        if kind == 'small':  # from A3C paper
            y = tf.nn.relu(U.conv2d(y, 16, "yl1", [3, 3], [1, 1], pad="SAME"))
            y = tf.nn.relu(U.conv2d(y, 32, "yl2", [3, 3], [1, 1], pad="SAME"))
            y = U.flattenallbut0(y)
            y = tf.nn.relu(U.dense(y, 256, 'ylin', U.normc_initializer(1.0)))
        elif kind == 'large':  # Nature DQN
            y = tf.nn.relu(U.conv2d(y, 64, "yl1", [3, 3], [1, 1], pad="SAME"))
            y = tf.nn.relu(U.conv2d(y, 64, "yl2", [3, 3], [1, 1], pad="SAME"))
            y = tf.nn.relu(U.conv2d(y, 128, "yl3", [3, 3], [1, 1], pad="SAME"))
            y = U.flattenallbut0(y)
            y = tf.nn.relu(U.dense(y, 512, 'ylin', U.normc_initializer(1.0)))
        else:
            raise NotImplementedError
        '''

        #logits = tf.reshape(c_caps2, (cfg.batch_size, -1))
        logits = U.dense(x, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = U.dense(y, 1, "value", U.normc_initializer(1.0))[:, 0]
        #self.vpred = caps2 * 100
        #U.dense(y, 1, "value", U.normc_initializer(1.0))[:,0]

        orgin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ob = np.tile(ob[np.newaxis, :], [cfg.batch_size, 1, 1, 1])
        ac1, vpred1 =  self._act(stochastic, ob)
        '''
        print(_2[0])
        print(ac1[0])
        print(vpred1[0])
        print('......')
        time.sleep(3)
        '''
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

