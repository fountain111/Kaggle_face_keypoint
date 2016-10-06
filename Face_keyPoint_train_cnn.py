from Face_kyePoint_cnn import *
from Global_defintion import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 25000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('summary_dir', '/tmp/faceKeypoint',
                           """Directory where to write event logs """
                           """and checkpoint.""")


def train():
    train_datas, train_labels, validation_datas, validation_labels = split_data()
    epochs_completed = 0
    index_in_epoch = 0
    num_examples = train_datas.shape[0]
    with tf.Graph().as_default():
        x = tf.placeholder('float', shape=[None,  INPUT_SIZE])
        y_ = tf.placeholder('float', shape=[None,  LABEL_SIZE])
        keep_prob = tf.placeholder('float')

        model_labels = inference(x, keep_prob)

        loss_op = loss(model_labels, y_)

        optimizer = train_in_cnn(loss_op)

        tf.scalar_summary("loss", loss_op)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        merge = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/train', sess.graph)
        validation_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/validation')

    for step in range(FLAGS.max_steps):
        batch_x, batch_y, index_in_epoch, epochs_completed,train_datas,train_labels = \
            get_batch(BATCH_SIZE, train_datas, train_labels, index_in_epoch, epochs_completed, num_examples)
        sess.run([optimizer,merge], feed_dict={x: batch_x, y_: batch_y,keep_prob: 0.5})
        train_writer.add_summary(sess.run(merge, feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0}), step)
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            validation_writer.add_summary(sess.run(merge, feed_dict={x: validation_datas, y_: validation_labels,keep_prob: 1.0}), step)
        if step == (FLAGS.max_steps - 1):
            print (epochs_completed)






def main(argv=None):
    train()



if __name__ == '__main__':
  tf.app.run()