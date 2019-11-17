import tensorflow as tf
import util.OPTS as OPTS
import numpy as np
from util.PyMatData import PyMatData
import os.path


class VGG_Face:
    class OPTS(OPTS.OPTS):
        def __init__(self):
             OPTS.OPTS.__init__(self, 'VGG_Face OPTS')
             self.network_name = None
             self.image_mean = np.reshape([129.1863, 104.7624, 93.5940], [1, 1, 1, 3])
             self.weight_path = None
             self.apply_dropout = None

    def __init__(self, opts):
        if opts is None:
            opts = self.OPTS()
        self.opts = opts
        self.opts.assert_all_keys_valid()

    def normalize_input(self, x):
        return x - self.opts.image_mean

    def construct(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input_image')
        self.x_normalized = self.normalize_input(self.x)
        self.network(self.x_normalized)

    def network(self, x):
        self.keep_prob = tf.placeholder(tf.float32)
        self.pool1 = self.vgg_conv(x, dim=64, num_conv=2, layer_num=1)
        self.pool2 = self.vgg_conv(self.pool1, dim=128, num_conv=2, layer_num=2)
        self.pool3 = self.vgg_conv(self.pool2, dim=256, num_conv=3, layer_num=3)
        self.pool4 = self.vgg_conv(self.pool3, dim=512, num_conv=3, layer_num=4)
        self.pool5 = self.vgg_conv(self.pool4, dim=512, num_conv=3, layer_num=5)
        #self.pool5 = self.vgg_conv_cbam(self.pool4, dim=512, num_conv=3, layer_num=5)
        self.fc6 = tf.nn.relu(self.fc(self.pool5, 4096, 'fc6'))
        if self.opts.apply_dropout:
            self.fc6 = tf.nn.dropout(self.fc6, self.keep_prob, name='fc6_drop')
        self.fc7 = tf.nn.relu(self.fc(self.fc6, 4096, 'fc7'))
        if self.opts.apply_dropout:
            self.fc7 = tf.nn.dropout(self.fc7, self.keep_prob, name='fc7_drop')
        self.fc8 = self.fc(self.fc7, 2622, 'fc8')
        self.softmax = tf.nn.softmax(self.fc8, name='softmax')

    def load_pretrained(self, session):
        if not os.path.isfile(self.opts.weight_path):
            raise OSError(2, 'No such file or directory', self.opts.weight_path)
        mat_data = PyMatData(self.opts.weight_path)

        for l in range(len(mat_data.layers)):
            layer = mat_data.layers[l]
            if layer['type'] == 'conv':
                with tf.variable_scope(layer['name'], reuse=True):
                    try:
                        weights = tf.get_variable('weights')
                        biases = tf.get_variable('biases')
                        if len(weights.get_shape()) == 2:
                            w_data = np.reshape(layer['weights'][0], [-1, weights.get_shape().as_list()[-1]])
                            b_data = layer['weights'][1]
                        else:
                            w_data = layer['weights'][0]
                            b_data = layer['weights'][1]
                        session.run(weights.assign(w_data))
                        session.run(biases.assign(b_data))
                    except ValueError:
                        print ("[Load Pretrained] layer \"%s\" unused" % layer['name'])
                        pass

    def fc(self, x, dim, name, reuse=False):
        in_shape = x.get_shape()
        s = 1
        for i in range(1, len(in_shape)):
            s *= int(in_shape[i])
        if len(in_shape) >= 4:
            x = tf.reshape(x, [-1, s])
        with tf.variable_scope(name, reuse=reuse):
            weights = tf.get_variable("weights", shape=[s, dim], initializer=tf.contrib.layers.xavier_initializer())
            biases = tf.get_variable("biases", shape=[dim], initializer=tf.constant_initializer(0.0))
            fc = tf.nn.xw_plus_b(x, weights, biases)
            return fc

    def vgg_conv_cbam(self, x, dim, num_conv=3, layer_num=None):
        t = x
        for i in range(1, num_conv + 1):
            t = self.vgg_conv2d(t, dim, "conv%d_%d" % (layer_num, i))
            t = cbam_block(t, "cbam%d_%d" % (layer_num, i), 8)
        t = self.vgg_pool2d(t, "pool%d" % (layer_num))
        return t
        
    def vgg_conv(self, x, dim, num_conv=3, layer_num=None):
        t = x
        for i in range(1, num_conv + 1):
            t = self.vgg_conv2d(t, dim, "conv%d_%d" % (layer_num, i))
        t = self.vgg_pool2d(t, "pool%d" % (layer_num))
        return t

    def vgg_conv2d(self, x, dim, name, reuse=False, trainable=True):
        in_shape = x.get_shape().as_list()
        with tf.variable_scope(name, reuse=reuse):
            weights = tf.get_variable("weights", shape=[3, 3, in_shape[-1], dim], initializer=tf.contrib.layers.xavier_initializer(), trainable=trainable)
            biases = tf.get_variable("biases", shape=[dim], initializer=tf.constant_initializer(0.0), trainable=trainable)
            conv = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
            return tf.nn.relu(conv + biases)

    def vgg_pool2d(self, in_tensor, name, reuse=False):
        with tf.name_scope(name):
            pool = tf.nn.max_pool(in_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            return pool

