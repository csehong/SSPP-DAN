import tensorflow as tf
import os
from DAN import Dom_Adapt_Net as Network



# Set Flag for Experiment
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', 'expr/F3D_30_60_FC6_FC6', 'Directory containing summary information about the experiment')
flags.DEFINE_integer('test_batch_size', 100, 'Test batch size')


# Set Domain Adaptation Network
net_opts = Network.OPTS()
net_opts.network_name = 'dom_adapt_net'
net_opts.weight_path = 'pretrained/vgg-face.mat'
net_opts.num_class = 30
net = Network(net_opts)
net.construct()
net.probe_prob = tf.nn.softmax(net.class_score)
net.probe_correct_pred = tf.equal(tf.argmax(net.probe_prob[:,0:net.opts.num_class], 1), tf.argmax(net.y_, 1))
net.probe_class_accuracy = tf.reduce_mean(tf.cast(net.probe_correct_pred, tf.float32))


# Set Dataset Manager
from data.data_manager import Manager as DataManager
dataset = DataManager('./data', net_opts.num_class)

# Start Session
saver = tf.train.Saver(tf.global_variables())
with tf.Session() as sess:
    # Load Pretrained Model (VGG-Face)
    sess.run(tf.global_variables_initializer())
    net.vgg_net.load_pretrained(sess)

    # Set Writier, Checkpoint folder
    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
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
    test_batch = dataset.eval_batch_generator(FLAGS.test_batch_size)

    # Run Session
    acc_loss, acc_d_loss, acc_c_loss, acc_d_acc, acc_c_acc = 0., 0., 0., 0., 0.
    acc_b = 0
    num_left = 1
    while num_left > 0:
        x_batch, y_batch, idx, dom_label, num_left = test_batch.next()
        b_size = x_batch.shape[0]
        loss, d_loss, c_loss, d_acc, c_acc, embed_feat = sess.run([net.loss, net.dom_loss, net.class_loss, net.dom_accuracy, net.probe_class_accuracy, net.embedded_with_class],
                                                                  feed_dict={net.x: x_batch, net.y_: y_batch, net.d_: dom_label, net.with_class_idx: idx, net.keep_prob: 1., net.l: 1.})
        acc_loss += loss * b_size
        acc_d_loss += d_loss * b_size
        acc_c_loss += c_loss * b_size
        acc_d_acc += d_acc * b_size
        acc_c_acc += c_acc * b_size
        acc_b += b_size
    print("[AGGR. TEST iter %d] costs(a,d,c)=(%4.4g,%4.4g,%4.4g) dom_acc: %.6f, class_acc: %.6f\n" %(step, acc_loss / acc_b, acc_d_loss / acc_b, acc_c_loss / acc_b, acc_d_acc / acc_b, acc_c_acc / acc_b))



if __name__ == '__main__':
  tf.app.run()

