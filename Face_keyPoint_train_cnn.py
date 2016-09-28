import tensorflow as tf
import pandas as pd
import numpy as np
import Face_kyePoint_cnn
BATCH_SIZE = 50
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 2500,
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
        x = tf.placeholder('float', shape=[None, 9216])
        y_ = tf.placeholder('float', shape=[None, 30])
        keep_prob = tf.placeholder('float')

        model_labels,regularizers = Face_kyePoint_cnn.inference(x, keep_prob)

        loss = Face_kyePoint_cnn.loss(model_labels, y_, regularizers)

        optimizer = Face_kyePoint_cnn.train(loss)

        # evaluation
        correct_prediction = tf.equal(tf.argmax(model_labels, 1), tf.argmax(y_, 1))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))

        predict = tf.argmax(model_labels, 1)

        tf.scalar_summary("loss",loss)
        tf.scalar_summary("accuracy",accuracy)



        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        merge = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/train', sess.graph)
        validation_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/validation')

    for step in range(FLAGS.max_steps):
        batch_x, batch_y, index_in_epoch, epochs_completed = Face_kyePoint_cnn.get_batch(BATCH_SIZE, train_datas, train_labels, index_in_epoch, epochs_completed, num_examples)
        sess.run([optimizer], feed_dict={x: batch_x, y_: batch_y,keep_prob:0.5})
        if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
            validation_writer.add_summary(sess.run(merge, feed_dict={x: validation_datas[0:Face_kyePoint_cnn.VALIDATION_SIZE], y_: validation_labels[0:Face_kyePoint_cnn.VALIDATION_SIZE], keep_prob:1.0}), step)
            train_writer.add_summary(sess.run(merge,feed_dict={x: batch_x, y_: batch_y,keep_prob:1.0}), step)
        if step == FLAGS.max_steps:
           # tf.train.Saver(sess,'cnn_handwritten_model')
            print ("done")






def main(argv=None):
    train()



if __name__ == '__main__':
  tf.app.run()