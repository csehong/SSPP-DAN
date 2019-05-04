import tensorflow as tf
import numpy as np
import os
from DAN import Dom_Adapt_Net as Network
from util.Logger import Logger



# Set Flag for Experiment
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 1e-5, 'Initial learning rate.')
flags.DEFINE_float('keep_prob', 0.5, 'Dropout rate (for keeping)')
flags.DEFINE_integer('max_steps', 10000, 'Maximum number of steps for training')
flags.DEFINE_integer('batch_size', 256, 'Training Batch size')
flags.DEFINE_integer('test_batch_size', 256, 'Test batch size')
flags.DEFINE_integer('display_step', 50, 'Display step for training')
flags.DEFINE_integer('test_step', 50, 'Display step for test')
flags.DEFINE_string('summaries_dir', 'expr/dom_10__dr_0.5__lr_1e-5__FC7_FC7', 'Directory containing summary information about the experiment')

def main(_):
    # Set Domain Adaptation Network
    net_opts = Network.OPTS()
    net_opts.network_name = 'dom_adapt_net'
    net_opts.weight_path = 'pretrained/vgg-face.mat' #download link: http://www.vlfeat.org/matconvnet/models/vgg-face.mat
    net_opts.num_class = 30
    net = Network(net_opts)
    net.construct()


    # Set Dataset Manager
    from data.data_manager import Manager as DataManager
    dataset = DataManager('./data', net_opts.num_class)


    # Set Optimizer (fine-tuning VGG-Face)
    with tf.variable_scope('optimizer'):
        net.trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        net.trainable_var_names = [v.name for v in net.trainable_vars]
        to_select_names = ('fc7', 'dom', 'class')
        net.sel_vars = []
        for i in range(len(net.trainable_var_names)):
            if net.trainable_var_names[i].startswith(to_select_names) :
                net.sel_vars.append(net.trainable_vars[i])
        net.adam = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(net.loss, var_list=net.sel_vars)


    # Start Session
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
    with tf.Session() as sess:
        # Load Pretrained Model (VGG-Face)
        sess.run(tf.global_variables_initializer())
        net.vgg_net.load_pretrained(sess)

        # Set Writier, Logger, Checkpoint folder
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test', sess.graph)
        logger = Logger(FLAGS.summaries_dir)
        logger.write(str(FLAGS.__flags))
        checkpoint_dir = os.path.join(FLAGS.summaries_dir, 'checkpoints')
        checkpoint_prefix = os.path.join(checkpoint_dir, "model.ckpt")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Restore Checkpoint
        step = 0
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            step = int(ckpt.model_checkpoint_path.split('-')[-1])
            print('Session restored successfully. step: {0}'.format(step))
            step += 1

        # Generate Mini-batch
        train_batch = dataset.batch_generator_thread(FLAGS.batch_size, 'train')
        test_batch = dataset.batch_generator_thread(FLAGS.test_batch_size, 'test')

        # Run Session
        c_acc_max = -10000.0
        for i in range(step, FLAGS.max_steps):
            p = float(i) / (FLAGS.max_steps)
            lamb = (2. / (1. + np.exp(-50. * p)) - 1.)
            x_batch, y_batch, idx, dom_label = train_batch.next()
            sess.run(net.adam, feed_dict={net.x: x_batch, net.y_: y_batch, net.d_: dom_label, net.with_class_idx: idx, net.keep_prob: FLAGS.keep_prob, net.l: lamb})

            if (i + 1) % FLAGS.display_step == 0:
                loss, d_loss, c_loss, d_acc, c_acc = sess.run([net.loss, net.dom_loss, net.class_loss, net.dom_accuracy, net.class_accuracy],
                                                              feed_dict={net.x: x_batch, net.y_: y_batch, net.d_: dom_label, net.with_class_idx: idx,
                                                                         net.keep_prob: 1., net.l: lamb})
                logger.write("[iter %d] costs(a,d,c)=(%4.4g,%4.4g,%4.4g) dom_acc: %.6f, class_acc: %.6f" %(i + 1, loss, d_loss, c_loss, d_acc, c_acc))
                short_summary = tf.Summary(value=[
                    tf.Summary.Value(tag="loss/loss", simple_value=float(loss)),
                    tf.Summary.Value(tag="loss/dom", simple_value=float(d_loss)),
                    tf.Summary.Value(tag="loss/cat", simple_value=float(c_loss)),
                    tf.Summary.Value(tag="acc/dom", simple_value=float(d_acc)),
                    tf.Summary.Value(tag="acc/cat", simple_value=float(c_acc)),
                    tf.Summary.Value(tag="lambda", simple_value=float(lamb)),
                ])
                train_writer.add_summary(short_summary, i)

            if (i + 1) % FLAGS.test_step == 0:
                x_batch, y_batch, idx, dom_label = test_batch.next()
                loss, d_loss, c_loss, d_acc, c_acc = sess.run([net.loss, net.dom_loss, net.class_loss, net.dom_accuracy, net.class_accuracy],
                                                              feed_dict={net.x: x_batch, net.y_: y_batch, net.d_: dom_label, net.with_class_idx: idx,
                                                                         net.keep_prob: 1., net.l: lamb})
                if c_acc > c_acc_max:
                    saver.save(sess, checkpoint_prefix, global_step=i+1)
                    c_acc_max = c_acc

                logger.write("\n[Test iter %d] costs(a,d,c)=(%4.4g,%4.4g,%4.4g) dom_acc: %.6f, class_acc: %.6f (max: %.4f)" % (i + 1, loss, d_loss, c_loss, d_acc, c_acc, c_acc_max))
                short_summary_test = tf.Summary(value=[
			        tf.Summary.Value(tag="loss/loss", simple_value=float(loss)),
	                tf.Summary.Value(tag="loss/dom", simple_value=float(d_loss)),
	                tf.Summary.Value(tag="loss/cat", simple_value=float(c_loss)),
	                tf.Summary.Value(tag="acc/dom", simple_value=float(d_acc)),
	                tf.Summary.Value(tag="acc/cat", simple_value=float(c_acc)),
					tf.Summary.Value(tag="lambda", simple_value=float(lamb)),
                ])
                test_writer.add_summary(short_summary_test, i)

if __name__ == '__main__':
  tf.app.run()
