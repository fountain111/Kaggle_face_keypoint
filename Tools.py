import numpy as np
import pandas as pd
from Global_defintion import *

flip_indices = [
    (0, 2), (1, 3),
    (4, 8), (5, 9), (6, 10), (7, 11),
    (12, 16), (13, 17), (14, 18), (15, 19),
    (22, 24), (23, 25)]

def data_argument():
    datas = pd.read_csv('training.csv').dropna()
    images = np.vstack(datas['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0).values).astype(np.float32).reshape(-1, INPUT_SIZE)
    labels = datas[datas.columns[:-1]].values / 96
    images_flip = np.copy(images)
    images_flip = images_flip.reshape(-1, 1, 96, 96)
    images_flip = images_flip[:, :, :, ::-1]
    images_flip = images_flip.reshape(-1, INPUT_SIZE)
    labels_flip = np.copy(labels)
    for a, b in flip_indices:
        labels_flip[flip_indices, a], labels_flip[flip_indices, b] = (labels_flip[flip_indices,b],labels_flip[flip_indices, a])

    images = np.vstack((images_flip,images))
    labels = np.vstack((labels_flip,labels))
    #df = pd.DataFrame(images)
    #df.to_csv('images.csv')
    # split data into train&cross_validation
    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]
    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]
    return train_images, train_labels, validation_images, validation_labels

def shuffle(datas):
    perm = np.arange(datas.shape[0])
    np.random.shuffle(perm)
    return datas[perm]


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
    return train_images[start:end], train_labels[start:end], index_in_epoch, epochs_completed, train_images, train_labels

def flip_data():
    datas = pd.read_csv('training.csv').dropna()
    images_1 = np.vstack(datas['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0).values).astype(np.float32).reshape(-1, INPUT_SIZE)
    labels = datas[datas.columns[:-1]].values / 96
    images = images_1.reshape(-1,1,96,96)
    images = images[:,:,:,::-1]
    images = images.reshape(-1,INPUT_SIZE)

    # split data into train&cross_validation
    train_images = images[VALIDATION_SIZE:]
    train_labels = labels[VALIDATION_SIZE:]
    validation_images = images[:VALIDATION_SIZE]
    validation_labels = labels[:VALIDATION_SIZE]
    return train_images, train_labels, validation_images, validation_labels