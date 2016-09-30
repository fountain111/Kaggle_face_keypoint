import tensorflow as tf
import pandas as pd
import numpy as np

VALIDATION_SIZE = 400
INPUT_SIZE = 9216
LABEL_SIZE = 30
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('regular', '0', """if use regularization to prevent overfitting. 0 = not use""")
LEARNING_RATE = 1e-4


def split_data():
    datas = pd.read_csv('training.csv').dropna()
    images = np.vstack(datas['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0).values).astype(np.float32).reshape(-1, INPUT_SIZE)
   # images = np.vstack(images.values)
    #images = images.reshape(-1, 96, 96, 1)
    labels = (datas[datas.columns[:-1]].values - 48) / 48
    # split data into train&corss_validation
    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]
    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]

    return train_images, train_labels, validation_images, validation_labels


def inference(datas):
    #kernel_size1 = [3, 3, 1, 32]
    #kernel_size2 = [3, 3, 32, 64]

    w1 = weight_variable([INPUT_SIZE, 100])
    b1 = bias_variable([100])
    hidden = tf.nn.relu(tf.matmul(datas,w1)+b1)

    w2 = weight_variable([100, LABEL_SIZE])
    b2 = bias_variable([LABEL_SIZE])

    labels = tf.nn.softmax(tf.matmul(hidden, w2) + b2)

    if FLAGS.regular:
        regularizer = tf.nn.l2_loss(w2)+tf.nn.l2_loss(b2)
    else:
        regularizer = 0
    return labels, regularizer


def loss(model_labels, labels, regularizers):
    loss = tf.reduce_mean(tf.square(model_labels - labels))
   # loss = -tf.reduce_sum(labels*tf.log(model_labels))
    if regularizers != 0:
        loss += 5e-4 * regularizers
    return loss


def train(loss):
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    return train_step


# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def get_batch(batch_size, train_images, train_labels, index_in_epoch, epochs_completed, num_examples):
    start = index_in_epoch
    index_in_epoch += batch_size
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > num_examples:
        # finished epoch
        epochs_completed += 1
        # shuffle the data
        perm = np.arange(num_examples)
        np.random.shuffle(perm)
        train_images = train_images[perm]
        train_labels = train_labels[perm]
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= num_examples
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch, epochs_completed


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv(datas, parameter):
    return tf.nn.conv2d(datas, parameter, strides=[1, 1, 1, 1], padding='SAME')


