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
    images = np.vstack(datas['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0).values).astype(np.float32)#.reshape(-1, INPUT_SIZE)
    labels = datas[datas.columns[:-1]].values / 96
    # split data into train&cross_validation
    train_images = images[VALIDATION_SIZE:,...]
    train_labels = labels[VALIDATION_SIZE:]
    validation_images = images[:VALIDATION_SIZE,...]
    validation_labels = labels[:VALIDATION_SIZE]
    return train_images, train_labels, validation_images, validation_labels


def inference(datas, keep_prob):
    datas = tf.reshape(datas, [-1, 96, 96, 1])

    kernel_size1 = [5, 5, 1, 32]
    kernel_size2 = [5, 5, 32, 64]
    #kernel_size3 = [2, 2, 64, 128]


    conv1_w = weight_variable(kernel_size1)    #conv1_b= bias_variable([32])
    gamma_conv1 = weight_variable([32])
    beta_conv1 = weight_variable([32])
    bn_conv1 = conv_bn(datas,conv1_w,gamma_conv1,beta_conv1,EPSILON)
    pool_1 = max_pool_2x2(tf.nn.relu(bn_conv1))

    conv2_w = weight_variable(kernel_size2)
    gamma_conv2 = weight_variable([64])
    beta_conv2 = weight_variable([64])
    bn_conv2 = conv_bn(pool_1,conv2_w,gamma_conv2,beta_conv2,EPSILON)
    pool_2 = max_pool_2x2(tf.nn.relu(bn_conv2))

    #conv3_w = weight_variable(kernel_size3)
    #conv3_b = bias_variable([128])
    #pool_3 = tf.nn.dropout(max_pool_2x2(tf.nn.relu(conv(pool_2, conv3_w) + conv3_b)),keep_prob)

    fc1_w = weight_variable([96 // 4 * 96 // 4*64, 512])
    fc1_b= bias_variable([512])
    gamma_fc1 = weight_variable([512])
    beta_fc1 = weight_variable([512])
    pool_2 = tf.reshape(pool_2, [-1, 96 // 4 * 96 // 4*64])
    bn_fc1 = norm_bn(pool_2,fc1_w,fc1_b,gamma_fc1,beta_fc1,EPSILON)
    fc_1 = tf.nn.relu(bn_fc1)

    fc2_w = weight_variable([512, 512])
    fc2_b = bias_variable([512])
    gamma_fc2 = weight_variable([512])
    beta_fc2 = weight_variable([512])
    bn_fc2 = norm_bn(fc_1,fc2_w,fc2_b,gamma_fc2,beta_fc2,EPSILON)
    fc_2 = tf.nn.relu(bn_fc2)

    fc3_w = weight_variable([512, 30])
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


def conv_bn(datas,conv_w,gamma,beta,epsilon):
    conv_t = conv(datas,conv_w)
    batch_mean, batch_variance = tf.nn.moments(conv_t, list(range(len(conv_t.get_shape()) - 1)))
    return tf.nn.batch_normalization(conv_t, batch_mean, batch_mean, beta, gamma, epsilon)


def norm_bn(datas,weights,biases,gamma,beta,epsilon):
    layer = tf.matmul(datas, weights) +biases
    batch_mean, batch_variance = tf.nn.moments(layer, list(range(len(layer.get_shape()) - 1)))
    return tf.nn.batch_normalization(layer, batch_mean, batch_mean, beta, gamma, epsilon)
