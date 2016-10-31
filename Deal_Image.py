import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
INPUT_IMAGE_DIM = 9216
IMAGE_DIR = '/Users/kakurong/PycharmProjects/untitled1/IMG_2412.JPG'

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

