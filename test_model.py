import tensorflow as tf
import os
from DAN import Dom_Adapt_Net as Network



# Set Flag for Experiment
flags = tf.app.flags
flags.DEFINE_string('dataset', 'eklfh_s1', 'eklfh_s1, eklfh_s2, scface_s1, scface_s2' )
flags.DEFINE_string('exp_mode', 'dom_3D_cycle', 'lower, lower_3D, lower_3D_cycle, dom, dom_3D, dom_3D_cycle, semi, semi_3D, semi_3D_cycle, upper')
flags.DEFINE_string('emb_layer', 'fc7', 'fc6, fc7')
flags.DEFINE_string('summaries_dir', 'exp_eklfh_s1/tuning/exp_2_dom__batch_64__steps_10000__lr_2e-05__embfc7__dr_0.3__ft_fc7', 'Directory containing summary information about the experiment')
flags.DEFINE_integer('test_batch_size', 256, 'Test batch size')




def main(_):
    FLAGS = flags.FLAGS

    print(FLAGS.summaries_dir)
    print("test_batch_size: ", FLAGS.test_batch_size)
    # Set Dataset Manager
    from data.data_manager import Manager as DataManager
    dataset = DataManager('./data', FLAGS.dataset, FLAGS.exp_mode)

    # Set Domain Adaptation Network
    net_opts = Network.OPTS()
    net_opts.network_name = 'dom_adapt_net'
    net_opts.weight_path = 'pretrained/vgg-face.mat'
    net_opts.num_class = dataset.num_class
    net = Network(net_opts, FLAGS.emb_layer)
    net.construct()
    net.probe_prob = tf.nn.softmax(net.class_score)
    net.probe_correct_pred = tf.equal(tf.argmax(net.probe_prob[:,0:net.opts.num_class], 1), tf.argmax(net.y_, 1))
    net.probe_class_accuracy = tf.reduce_mean(tf.cast(net.probe_correct_pred, tf.float32))




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
            x_batch, y_batch, idx, dom_label, num_left = test_batch.__next__()
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

