import tensorflow as tf
import util.OPTS as OPTS
from pretrained.VGG_Face import VGG_Face
from util.flip_gradient import flip_gradient



# Domain Adaptation Model
class Dom_Adapt_Net:
    class OPTS(OPTS.OPTS):
        def __init__(self):
            OPTS.OPTS.__init__(self, 'VGG_Face OPTS')
            self.network_name = None
            self.weight_path = None
            self.num_class = None

    # Initialization
    def __init__(self, opts):
        if opts is None:
            opts = self.OPTS()
        self.opts = opts
        self.opts.assert_all_keys_valid()

        self.x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='input_image')
        self.with_class_idx = tf.placeholder(tf.int32, shape=[None], name='with_class_idx')
        self.l = tf.placeholder(tf.float32)
        self.keep_prob = None
        self.d_ = tf.placeholder(tf.float32, shape=[None, 2], name='input_domain_label')  # for 2 domains
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.opts.num_class], name='input_class_label') # for N classes

    # Construct Overall Framework
    def construct(self):
        net_opts = VGG_Face.OPTS()
        net_opts.network_name = 'vgg_face_net'
        net_opts.weight_path = 'pretrained/vgg-face.mat'
        net_opts.apply_dropout = True

        self.vgg_net = VGG_Face(net_opts)
        x_normalized = self.vgg_net.normalize_input(self.x)
        self.vgg_net.network(x_normalized)
        self.keep_prob = self.vgg_net.keep_prob
        self.embedded = self.vgg_net.fc6       #Fine tuning from FC6 of VGG-Face
        self.embedded_with_class = tf.gather(self.embedded, self.with_class_idx, name='embedded_with_class')
        self.dom_network(self.embedded)
        self.class_network(self.embedded_with_class)
        self.loss = self.dom_loss + self.class_loss

    # Construct Domain Discriminator Network
    def dom_network(self, x):
        x_flip = flip_gradient(x, self.l)
        fc1 = tf.nn.relu(self.fc(x_flip, 1024, 'dom_fc1'), name='dom_fc1_relu')
        fc1_drop = tf.nn.dropout(fc1, self.keep_prob)
        fc2 = tf.nn.relu(self.fc(fc1_drop, 1024, 'dom_fc2'), name='dom_fc2_relu')
        fc2_drop = tf.nn.dropout(fc2, self.keep_prob)

        self.dom_score = self.fc(fc2_drop, 2, 'dom_score')
        self.dom_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.dom_score, labels = self.d_))
        correct_pred = tf.equal(tf.argmax(self.dom_score, 1), tf.argmax(self.d_, 1))
        self.dom_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Construct Label Classifier Network
    def class_network(self, x):
        self.class_score = self.fc(x, self.opts.num_class, 'class_score')
        self.class_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = self.class_score, labels = self.y_))
        correct_pred = tf.equal(tf.argmax(self.class_score, 1), tf.argmax(self.y_, 1))
        self.class_accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Define Fully Connected Layer
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

