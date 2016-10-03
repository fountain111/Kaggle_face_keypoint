import tensorflow as tf
import pandas as pd
import numpy as np
import Face_kyePoint_cnn
BATCH_SIZE = 128
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 7000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('summary_dir', '/tmp/faceKeypoint',
                           """Directory where to write event logs """
                           """and checkpoint.""")



def train():
    train_datas, train_labels, validation_datas, validation_labels = Face_kyePoint_cnn.split_data()
    epochs_completed = 0
    index_in_epoch = 0
    num_examples = train_datas.shape[0]
    with tf.Graph().as_default():
        x = tf.placeholder('float', shape=[None, Face_kyePoint_cnn.INPUT_SIZE])
        y_ = tf.placeholder('float', shape=[None, Face_kyePoint_cnn.LABEL_SIZE])

        model_labels, regularizers = Face_kyePoint_cnn.inference(x)

        loss = Face_kyePoint_cnn.loss(model_labels, y_, regularizers)

        optimizer = Face_kyePoint_cnn.train(loss)

        tf.scalar_summary("loss",loss)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        merge = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/train', sess.graph)
        validation_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/validation')

    for step in range(FLAGS.max_steps):
        batch_x, batch_y, index_in_epoch, epochs_completed,train_datas,train_labels = Face_kyePoint_cnn.get_batch(BATCH_SIZE, train_datas, train_labels, index_in_epoch, epochs_completed, num_examples)
        sess.run([optimizer], feed_dict={x: batch_x, y_: batch_y})
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            validation_writer.add_summary(sess.run(merge, feed_dict={x: validation_datas[0:Face_kyePoint_cnn.VALIDATION_SIZE], y_: validation_labels[0:Face_kyePoint_cnn.VALIDATION_SIZE]}), step)
            train_writer.add_summary(sess.run(merge, feed_dict={x: batch_x, y_: batch_y}), step)
        if step == (FLAGS.max_steps - 1):
            print (epochs_completed)






def main(argv=None):
    train()



if __name__ == '__main__':
  tf.app.run()