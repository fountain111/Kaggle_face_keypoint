import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from utility import Tools as tl

INPUT_IMAGE_DIM = 9216
IMAGE_DIR = '/Users/kakurong/PycharmProjects/untitled1/IMG_2412.JPG'
flip_indices_eye_center = [
    (0, 2), (1, 3),
   ]


def get_image(sess):
    filename_queue = tf.train.string_input_producer([IMAGE_DIR])  # list of files to read
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)

    img = tf.image.decode_jpeg(value)
    # resized_image = tf.image.resize_images(my_img, 96, 96)
    # image_batch = tf.train.batch([my_img], batch_size=1)
    resized_image = tf.reshape(img, [1, INPUT_IMAGE_DIM])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    image = resized_image.eval()  # here is your image Tensor :)
    coord.request_stop()
    coord.join(threads)
    return image


def plot_sampleWithLables(input, labels,i,ifLables):
    img = input.reshape(96, 96)
    if ifLables:
        labels = pd.DataFrame(labels)
        colors = np.random.rand(2)
        area = np.pi * (10) ** 2  # 0 to 15 point radiuses
        #plt.scatter(x * 96, y * 96,hold=False)
        plt.scatter(labels.ix[:,0][i]*96,labels.ix[:,1][i]*96, s=area, alpha=0.5)
        plt.scatter(labels.ix[:,2][i]*96,labels.ix[:,3][i]*96,s= area)
    plt.imshow(img, cmap='gray')
   # plt.scatter(truth[0::2] * 96, truth[1::2] * 96, c='r', marker='x')
    plt.savefig("data/img"+ str(i) + ".png")
    plt.imshow(img, cmap='gray',hold=False)
    return


def flip_images_labels(images,labels,location):
    images_flip = np.copy(images)
    images_flip = images.reshape(-1, 1, 96, 96)
    images_flip = images_flip[:, :, :, ::-1]
    images_flip = images_flip.reshape(-1, INPUT_IMAGE_DIM)
    labels_flip = np.copy(labels)
    if location == "eye_center":
        labels_flip[:,0] = 1-labels[:, 2]
        labels_flip[:,1] = labels[:, 3]
        labels_flip[:,2] = 1-labels[:, 0]
        labels_flip[:,3] = labels[:, 1]
    return images_flip, labels_flip


def DataCenter_eye(input, start, end):
    images = np.vstack(input['Image'].apply(lambda im: np.fromstring(im, sep=' ') / 255.0).values).astype(
        np.float32)
    labels = input[input.columns[:-1]].values / 96
    images_flip,labels_flip = flip_images_labels(images,labels,"eye_center")

    #stack
    images_stack = np.vstack((images,images_flip))
    labels_stack = np.vstack((labels,labels_flip))

    #shuffle
    images, labels = tl.shuffle(images_stack, labels_stack)

    #for index
    images = pd.DataFrame(images)
    labels = pd.DataFrame(labels)
    i_labels = labels.ix[:, start:end].dropna()
    i_images = images.ix[i_labels.index]
    i_labels = i_labels.as_matrix()
    i_images = i_images.as_matrix()


    return tl.split_trainValidation(i_images, i_labels)




def nothing():
    return 