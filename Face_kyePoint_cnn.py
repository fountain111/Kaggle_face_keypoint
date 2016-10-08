import tensorflow as tf
import pandas as pd
from Global_defintion import *
from Tools import *

FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_string('loss_and_L2', 'lossandL2', """ set a collection name""")

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(1e-3, global_step*BATCH_SIZE, TRARIN_SIZE, 0.95,staircase=True)

def split_data():
    datas = pd.read_csv('training.csv').dropna()
    images = np.vstack(datas['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0).values).astype(np.float32).reshape(-1, INPUT_SIZE)
    labels = datas[datas.columns[:-1]].values / 96
    # split data into train&cross_validation
    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]
    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]
    return train_images, train_labels, validation_images, validation_labels


def inference(datas, keep_prob):
    kernel_size1 = [5, 5, 1, 32]
    kernel_size2 = [5, 5, 32, 64]
    #kernel_size3 = [2, 2, 64, 128]

    conv1_w = weight_variable(kernel_size1)
    conv1_b= bias_variable([32])
    pool_1 = max_pool_2x2(tf.nn.relu(conv(tf.reshape(datas, [-1, 96, 96, 1]), conv1_w)+ conv1_b))

    conv2_w = weight_variable(kernel_size2)
    conv2_b = bias_variable([64])
    pool_2 = max_pool_2x2(tf.nn.relu(conv(pool_1, conv2_w)+conv2_b))


    #conv3_w = weight_variable(kernel_size3)
    #conv3_b = bias_variable([128])
    #pool_3 = tf.nn.dropout(max_pool_2x2(tf.nn.relu(conv(pool_2, conv3_w) + conv3_b)),keep_prob)


    fc1_w = weight_variable([24*24*64, 500])
    fc1_b = bias_variable([500])
    fc_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(tf.reshape(pool_2, [-1, 24*24*64]), fc1_w) + fc1_b), keep_prob)

    fc2_w = weight_variable([500, 500])
    fc2_b = bias_variable([500])
    fc_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc_1, fc2_w) + fc2_b), keep_prob)

    fc3_w = weight_variable([500, 30])
    fc3_b = bias_variable([30])
    labels = tf.matmul(fc_2, fc3_w) + fc3_b

    return labels


def loss(model_labels, labels):
    loss_value = tf.reduce_mean(tf.reduce_sum(tf.square(model_labels - labels), 1))
    #tf.add_to_collection(FLAGS.loss_and_L2, loss)
   # loss = -tf.reduce_sum(labels*tf.log(model_labels))
    return loss_value


def train_in_cnn(loss_op):
    train_step = tf.train.AdamOptimizer(learning_rate,0.95).minimize(loss_op, global_step=global_step)
    return train_step


# weight initialization with decay
def weight_variable(shape, wd):
    initial = tf.truncated_normal(shape, stddev=0.1)
    if wd:
        weight_decay = tf.mul(tf.nn.l2_loss(initial), wd)
        tf.add_to_collection(FLAGS.loss_and_L2, weight_decay)
    return tf.Variable(initial)

# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)





def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv(datas, parameter):
    return tf.nn.conv2d(datas, parameter, strides=[1, 1, 1, 1], padding='SAME')


