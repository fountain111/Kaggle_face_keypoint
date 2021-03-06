import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from Global_defintion import *
FLOOKUP = 'IdLookupTable.csv'

IMAGE_INPUT_WIDTH = 96
IMAGE_INPUT_HEIGH = 96



flip_indices_eye_center = [
    (0, 2), (1, 3),
    (4, 8), (5, 9), (6, 10), (7, 11),
    (12, 16), (13, 17), (14, 18), (15, 19),
    (22, 24), (23, 25)]

def data_argument():
    datas = pd.read_csv('training.csv').dropna()
    images,labels = flip_images(datas)
    images,labels = shuffle(images,labels)
    return split_trainValidation(images,labels)

def DataCenter_eye(input,start,end):
    images = np.vstack(input['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0).values).astype(
        np.float32)
    labels = input[input.columns[:-1]].values / 96

    images,labels = shuffle(images,labels)
    images = pd.DataFrame(images)
    labels = pd.DataFrame(labels)
    i_labels = labels.ix[:, start:end].dropna()
    i_images = images.ix[i_labels.index]
    i_labels = i_labels.as_matrix()
    i_images = i_images.as_matrix()
    return split_trainValidation(i_images, i_labels)


def dataArgument_withColumn(datas,start,end):
    images,labels = flip_images(datas)
    images,labels = shuffle(images,labels)
    images = pd.DataFrame(images)
    labels = pd.DataFrame(labels)
    i_labels = labels.ix[:,start:end].dropna()
    i_images = images.ix[i_labels.index]
    i_labels = i_labels.as_matrix()
    i_images = i_images.as_matrix()
    return split_trainValidation(i_images,i_labels)




def flip_images(input):
    images = np.vstack(input['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0).values).astype(
        np.float32)
    labels = input[input.columns[:-1]].values / 96
    images_flip = np.copy(images)
    images_flip = images_flip.reshape(-1, 1, 96, 96)
    images_flip = images_flip[:, :, :, ::-1]
    images_flip = images_flip.reshape(-1, INPUT_SIZE)
    labels_flip = np.copy(labels)
    for a, b in flip_indices_eye_center:
        labels_flip[flip_indices_eye_center, a], labels_flip[flip_indices_eye_center, b] = (
            labels_flip[flip_indices_eye_center, b], labels_flip[flip_indices_eye_center, a])

    images = np.vstack((images_flip, images))
    labels = np.vstack((labels_flip, labels))
    return images,labels

def split_trainValidation(input,labels):

    validation_size = np.int64(input[0].shape[0] * 0.05)
    train_images = input[validation_size:]
    train_labels = labels[validation_size:]
    validation_images = input[:validation_size]
    validation_labels = labels[:validation_size]
    return train_images,train_labels,validation_images,validation_labels


def shuffle(datas,labels):
    shuffle_index = np.arange(datas.shape[0])
    np.random.shuffle(shuffle_index)
    return datas[shuffle_index],labels[shuffle_index]


def get_batch(batch_size, index_in_epoch, epochs_completed,train_size,inputs,labels):
    start = index_in_epoch
    index_in_epoch += batch_size
    #those two variables for a new epoch  with shuufle.
    inputs_new = None
    labels_new = None
    # when all trainig data have been already used, it is reorder randomly
    if index_in_epoch > train_size:
        # finished epoch
        epochs_completed += 1
        print ("epoch =", epochs_completed)
        # shuffle the data
        #perm = np.arange(train_size)
        #np.random.shuffle(perm)
        #train_images = train_images[perm]
        #train_labels = train_labels[perm]
        inputs_new,labels_new =  shuffle(inputs, labels)
        inputs,labels = inputs_new,labels_new
        # start next epoch
        start = 0
        index_in_epoch = batch_size
        assert batch_size <= train_size
    end = index_in_epoch
    return inputs[start:end],labels[start:end], index_in_epoch, epochs_completed,inputs_new,labels_new

def next_batch(batch_size, train_images, train_labels, index_in_epoch, epochs_completed, train_size):
    start = index_in_epoch
    index_in_epoch += batch_size
    if index_in_epoch > train_size:
        epochs_completed += 1
        train_images,train_labels = shuffle(train_images,train_labels)
        start = 0
        index_in_epoch = batch_size
    end = index_in_epoch
    return train_images[start:end], train_labels[start:end], index_in_epoch, epochs_completed, train_images, train_labels


def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def batch_norm(datas, beta_and_gamma_size):
    beta = tf.Variable(tf.constant(0.0, shape=[beta_and_gamma_size]))

    gamma = tf.Variable(tf.constant(1.0, shape=[beta_and_gamma_size]))

    batch_mean, batch_var = tf.nn.moments(datas, [0, 1, 2])

    norm = tf.nn.batch_normalization(datas,mean,var,beta,gamma,1e-3)

def generate_submission(test_dataset, cols,sess, eval_prediction, x,keep_prob,phase_train):
    labels = eval_in_batches(test_dataset, sess, eval_prediction, x,keep_prob,phase_train)

    labels *= 96.0
    labels = labels.clip(0,96)
    labels = pd.DataFrame(labels)

    lookup_table = pd.read_csv(FLOOKUP)
    values = []
    #labels.to_csv('temp.csv', index=False)

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            labels.ix[row.ImageId - 1][np.where(cols == row.FeatureName)[0][0]],
        ))
    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv('submission.csv', index=False)


def eval_in_batches(data, sess, eval_prediction, x,keep_prob,phase_train):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    predictions = np.ndarray(shape=(size, LABEL_SIZE), dtype=np.float32)
    for begin in range(0, size, BATCH_SIZE):
        end = begin + BATCH_SIZE
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={x: data[begin:end, ...],keep_prob:1,phase_train:False})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={x: data[-BATCH_SIZE:, ...],keep_prob:1,phase_train:False})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions