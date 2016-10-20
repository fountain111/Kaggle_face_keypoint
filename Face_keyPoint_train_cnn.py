from Face_kyePoint_cnn import *
from Global_defintion import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('max_steps', 20000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_string('summary_dir', '/tmp/faceKeypoint',
                           """Directory where to write event logs """
                           """and checkpoint.""")


def train(if_train):
        train_datas, train_labels, validation_datas, validation_labels = split_data()
        epochs_completed = 0
        index_in_epoch = 0
        num_examples = train_datas.shape[0]
        #with tf.Graph().as_default():
        x = tf.placeholder('float', shape=[None,  INPUT_SIZE])
        y_ = tf.placeholder('float', shape=[None,  LABEL_SIZE])
        keep_prob = tf.placeholder('float')
        phase_train = tf.placeholder(tf.bool,name='phase_train')
        model_labels = inference_with_bn(x, keep_prob,phase_train)
        #model_labels = inference(x,keep_prob)

        loss_op = loss(model_labels, y_)

        optimizer = train_in_cnn(loss_op)

        saver = tf.train.Saver()
        tf.scalar_summary("loss", loss_op)
        init = tf.initialize_all_variables()
        sess = tf.InteractiveSession()
        sess.run(init)
        merge = tf.merge_all_summaries()
        train_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/train', sess.graph)
        validation_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/validation')
        image_op = tf.image_summary('x-input',tf.reshape(x,[-1,96,96,1]),max_images=BATCH_SIZE)
        image_writer = tf.train.SummaryWriter(FLAGS.summary_dir + '/images')
        best_valid = np.inf
        if if_train:
            for step in range(FLAGS.max_steps):
                batch_x, batch_y, index_in_epoch, epochs_completed,train_datas,train_labels = get_batch(BATCH_SIZE, train_datas, train_labels, index_in_epoch, epochs_completed, num_examples)
                sess.run([optimizer,merge,image_op], feed_dict={x: batch_x, y_: batch_y,keep_prob: 0.5,phase_train:True})
                train_writer.add_summary(sess.run(merge,feed_dict={x: batch_x, y_: batch_y, keep_prob: 1.0,phase_train:True}), step)
                validation_writer.add_summary(sess.run(merge,feed_dict={x: validation_datas, y_: validation_labels, keep_prob: 1.0,phase_train:False}), step)
                loss_valid = sess.run(loss_op,feed_dict={x: validation_datas, y_: validation_labels, keep_prob: 1.0,phase_train:False})
                image_writer.add_summary(sess.run(image_op,feed_dict={x:batch_x}))
                print (step)
                if loss_valid < best_valid:
                    best_valid = loss_valid
                    best_valid_step = step
                elif best_valid_step + EARY_STOP_PATIENCE < step:
                    print ("early stop at {},best loss was {}",step,best_valid)
            saver.save(sess,'model.ckpt',global_step=step+1)
        else:
            load_model = saver.restore(sess,'model.ckpt')
            test_data,cols = load_test_data()
            generate_submission(test_data,cols,sess,model_labels,x,keep_prob,phase_train)






def main(argv=None):
    train(if_train=False)




if __name__ == '__main__':
  tf.app.run()