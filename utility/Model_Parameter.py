import  tensorflow as tf
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

