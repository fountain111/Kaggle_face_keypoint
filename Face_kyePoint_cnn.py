import tensorflow as tf
import pandas as pd
from Global_defintion import *
from Tools import *

FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_string('loss_and_L2', 'lossandL2', """ set a collection name""")

global_step = tf.Variable(0)
learning_rate = tf.train.exponential_decay(1e-3, global_step*BATCH_SIZE, TRARIN_SIZE, 0.95,staircase=True)
tf.scalar_summary("learning_rate", learning_rate)

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


def load_test_data():
    datas = pd.read_csv('test.csv')
    train_data = pd.read_csv('training.csv')
    cols = train_data.columns[:-1]

    images = np.vstack(datas['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0).values).astype(np.float32)#.reshape(-1, INPUT_SIZE)
    return images,cols



def inference(datas, keep_prob):
    datas = tf.reshape(datas, [-1, 96, 96, 1])

    kernel_size1 = [5, 5, 1, 32]
    kernel_size2 = [5, 5, 32, 64]
    #kernel_size3 = [2, 2, 64, 128]


    conv1_w = weight_variable(kernel_size1)
    conv1_b= bias_variable([32])
    pool_1 = max_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv(datas,conv1_w),conv1_b)))

    conv2_w = weight_variable(kernel_size2)
    conv2_b = bias_variable([64])
    pool_2 = max_pool_2x2(tf.nn.relu(tf.nn.bias_add(conv(pool_1,conv2_w),conv2_b)))


    fc1_w = weight_variable([96 // 4 * 96 // 4*64, 512])
    fc1_b= bias_variable([512])
    pool_2 = tf.reshape(pool_2,[-1,96 // 4 * 96 // 4*64])
    fc_1 = tf.nn.dropout(tf.nn.relu(tf.matmul(pool_2,fc1_w) + fc1_b),keep_prob)

    fc2_w = weight_variable([512, 512])
    fc2_b = bias_variable([512])
    fc_2 = tf.nn.dropout(tf.nn.relu(tf.matmul(fc_1,fc2_w) + fc2_b),keep_prob)


    fc3_w = weight_variable([512, 30])
    fc3_b = bias_variable([30])
    labels = tf.matmul(fc_2, fc3_w) + fc3_b

    return labels

def inference_with_bn(datas, keep_prob,is_training):
    datas = tf.reshape(datas, [-1, 96, 96, 1])

    kernel_size1 = [5, 5, 1, 32]
    kernel_size2 = [5, 5, 32, 64]
    kernel_size3 = [5, 5, 64, 128]
    kernel_size4 = [5, 5, 128, 256]
    kernel_size5 = [5, 5, 256, 512]


    conv1_w = weight_variable(kernel_size1)
    conv1_b = bias_variable([32])
    conv1 = tf.nn.bias_add(conv(datas,conv1_w),conv1_b)
   # mean ,var = tf.nn.moments(conv1,[0,1,2])
   # bn_conv1 = tf.nn.batch_normalization(conv1,mean,var,conv1_beta,conv1_gamma,EPSILON)
    bn_conv1 = norm_layer(conv1,is_training)
    pool_1 = max_pool_2x2(tf.nn.relu(bn_conv1))

    conv2_w = weight_variable(kernel_size2)
    conv2_b = bias_variable([64])
    conv2 = tf.nn.bias_add(conv(pool_1, conv2_w), conv2_b)
    #mean, var = tf.nn.moments(conv2, [0, 1, 2])
    #bn_conv2 = tf.nn.batch_normalization(conv2, mean, var, conv2_beta, conv2_gamma, EPSILON)
    bn_conv2 = norm_layer(conv2,is_training)
    pool_2 = max_pool_2x2(tf.nn.relu(bn_conv2))

    conv3_w = weight_variable(kernel_size3)
    conv3_b = bias_variable([128])
    conv3 = tf.nn.bias_add(conv(pool_2, conv3_w), conv3_b)
    # mean, var = tf.nn.moments(conv2, [0, 1, 2])
    # bn_conv2 = tf.nn.batch_normalization(conv2, mean, var, conv2_beta, conv2_gamma, EPSILON)
    bn_conv3 = norm_layer(conv3, is_training)
    pool_3 = max_pool_2x2(tf.nn.relu(bn_conv3))

    conv4_w = weight_variable(kernel_size4)
    conv4_b = bias_variable([256])
    conv4 = tf.nn.bias_add(conv(pool_3, conv4_w), conv4_b)
    # mean, var = tf.nn.moments(conv2, [0, 1, 2])
    # bn_conv2 = tf.nn.batch_normalization(conv2, mean, var, conv2_beta, conv2_gamma, EPSILON)
    bn_conv4 = norm_layer(conv4, is_training)
    pool_4 = max_pool_2x2(tf.nn.relu(bn_conv4))

    conv5_w = weight_variable(kernel_size5)
    conv5_b = bias_variable([512])
    conv5 = tf.nn.bias_add(conv(pool_4, conv5_w), conv5_b)
    # mean, var = tf.nn.moments(conv2, [0, 1, 2])
    # bn_conv2 = tf.nn.batch_normalization(conv2, mean, var, conv2_beta, conv2_gamma, EPSILON)
    bn_conv5 = norm_layer(conv5, is_training)
    pool_5 = max_pool_2x2(tf.nn.relu(bn_conv5))

    scope = "fc1"
    pool_6 = tf.reshape(pool_5, [-1, 96 // 32 * 96 // 32*512])
    fc1_w = weight_variable([96 // 32 * 96 //32*512, 512])
    fc1_b= bias_variable([512])
    fc1_z = tf.matmul(pool_6,fc1_w) + fc1_b
    #mean, var = tf.nn.moments(fc1_z, [0])
    #bn_fc1 = tf.nn.batch_normalization(fc1_z,mean,var,fc1_beta,fc1_gamma,EPSILON)
    bn_fc1 = norm_layer(fc1_z,is_training)
    fc_1 = tf.nn.dropout(tf.nn.relu(bn_fc1),keep_prob)

    scope ="fc2"
    fc2_w = weight_variable([512, 512])
    fc2_b = bias_variable([512])
    fc2_z = tf.matmul(fc_1,fc2_w)+ fc2_b
    #mean, var = tf.nn.moments(fc2_z, [0])
    #bn_fc2 = tf.nn.batch_normalization(fc2_z,mean,var,fc2_beta,fc2_gamma,EPSILON)
    bn_fc2 = norm_layer(fc2_z,is_training)
    fc_2 = tf.nn.dropout(tf.nn.relu(bn_fc2),keep_prob)

    scope = "fc3"
    fc3_w = weight_variable([512, 4])
    fc3_b = bias_variable([4])
    labels = tf.matmul(fc_2, fc3_w) + fc3_b

    return labels


def loss(model_labels, labels):
    loss_value = tf.reduce_mean(tf.reduce_sum(tf.square(model_labels - labels), 1))
    #tf.add_to_collection(FLAGS.loss_and_L2, loss)
   # loss = -tf.reduce_sum(labels*tf.log(model_labels))
    return loss_value


def train_in_cnn(loss_op):
    train_step = tf.train.AdamOptimizer(learning_rate,0.98).minimize(loss_op, global_step=global_step)
    #train_step = tf.train.AdamOptimizer(1e-3).minimize(loss_op)

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



def norm_layer(inputs,phase_train,scope = None):
    #return tf.cond(phase_train,
     #   lambda:tf.contrib.layers.batch_norm(inputs,scale=True,updates_collections=None,scope=scope, is_training = True),

      #  lambda: tf.contrib.layers.batch_norm(inputs,scale=True,updates_collections=None,scope=scope,is_training=False))
    if phase_train is not None:
        return tf.contrib.layers.batch_norm(inputs, scale=True, updates_collections=None, is_training=True)
    else:
        return tf.contrib.layers.batch_norm(inputs, scale=True, updates_collections=None,
                                             is_training=False)





