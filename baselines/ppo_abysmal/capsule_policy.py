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

        epsilon = 1e-8

        with tf.variable_scope('actor'):
            with tf.variable_scope('Conv1_layer'):
                # Conv1, [batch_size, 20, 20, 32]
                conv1 = tf.contrib.layers.conv2d(ob, num_outputs=32,
                                                 kernel_size=9, stride=1,
                                                 padding='VALID')

            # Primary Capsules layer, return [batch_size, 288, 8, 1]
            with tf.variable_scope('PrimaryCaps_layer'):
                primaryCaps1 = CapsLayer(num_outputs=8, vec_len=8, with_routing=False, layer_type='CONV')
                caps1 = primaryCaps1(conv1, kernel_size=9, stride=2)

            # DigitCaps layer, return [batch_size, 10, 16, 1]
            with tf.variable_scope('DigitCaps_layer1'):
                digitCaps1 = CapsLayer(num_outputs=1, vec_len=1, with_routing=True, layer_type='FC')
                caps2 = digitCaps1(caps1)

        with tf.variable_scope('critic'):
            with tf.variable_scope('Conv1_layer'):
                # Conv1, [batch_size, 20, 20, 32]
                c_conv1 = tf.contrib.layers.conv2d(ob, num_outputs=32,
                                                 kernel_size=9, stride=1,
                                                 padding='VALID')

            # Primary Capsules layer, return [batch_size, 288, 8, 1]
            with tf.variable_scope('PrimaryCaps_layer'):
                c_primaryCaps1 = CapsLayer(num_outputs=8, vec_len=8, with_routing=False, layer_type='CONV')
                c_caps1 = c_primaryCaps1(c_conv1, kernel_size=9, stride=2)

            # DigitCaps layer, return [batch_size, 10, 16, 1]
            with tf.variable_scope('DigitCaps_layer1'):
                c_digitCaps1 = CapsLayer(num_outputs=3, vec_len=1, with_routing=True, layer_type='FC')
                c_caps2 = c_digitCaps1(c_caps1)

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

        logits = tf.reshape(c_caps2, (cfg.batch_size, -1))
        #U.dense(c_caps3, pdtype.param_shape()[0], "logits", U.normc_initializer(0.01))
        self.pd = pdtype.pdfromflat(logits)
        self.vpred = caps2 * 100
        #U.dense(y, 1, "value", U.normc_initializer(1.0))[:,0]

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = self.pd.sample() # XXX
        self._act = U.function([stochastic, ob], [ac, self.vpred, caps2, logits])

    def act(self, stochastic, ob):
        ob = np.tile(ob[np.newaxis, :], [cfg.batch_size, 1, 1, 1])
        ac1, vpred1, _1, _2 =  self._act(stochastic, ob)
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

