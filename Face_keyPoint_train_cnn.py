from Face_kyePoint_cnn import *
from Global_defintion import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 5000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('summary_dir', '/tmp/faceKeypoint',
                           """Directory where to write event logs """
                           """and checkpoint.""")


def train():
    train_datas, train_labels, validation_datas, validation_labels = split_data()
    epochs_completed = 0
    index_in_epoch = 0
    num_examples = train_datas.shape[0]
    #with tf.Graph().as_default():
    x = tf.placeholder('float', shape=[None,  INPUT_SIZE])
    y_ = tf.placeholder('float', shape=[None,  LABEL_SIZE])
    keep_prob = tf.placeholder('float')

    model_labels = inference_with_bn(x, keep_prob,is_training=True)
    #model_labels = inference(x,keep_prob)

    loss_op = loss(model_labels, y_)

    optimizer = train_in_cnn(loss_op)

    test_predicate = inference_with_bn(x,keep_prob,is_training=False)

    tf.scalar_summary("loss", loss_op)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    merge = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/train', sess.graph)
    validation_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/validation')
    image_op = tf.image_summary('x-input',tf.reshape(x,[-1,96,96,1]),max_images=BATCH_SIZE)
    image_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/images')

    for step in range(FLAGS.max_steps):
        batch_x, batch_y, index_in_epoch, epochs_completed,train_datas,train_labels = get_batch(BATCH_SIZE, train_datas, train_labels, index_in_epoch, epochs_completed, num_examples)
        sess.run([optimizer,merge,image_op], feed_dict={x: batch_x, y_: batch_y,keep_prob: 0.5})
        train_writer.add_summary(sess.run(merge,feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0}), step)
        validation_writer.add_summary(sess.run(merge, feed_dict={x: validation_datas, y_: validation_labels, keep_prob: 1.0}), step)
        image_writer.add_summary(sess.run(image_op,feed_dict={x:batch_x}))
    if early_stop:
        test_predicates = sess.run(test_predicate,feed_dict={x:test,keep_prob:1.0})
        to_csv(test_predicates)









def main(argv=None):
    train()



if __name__ == '__main__':
  tf.app.run()