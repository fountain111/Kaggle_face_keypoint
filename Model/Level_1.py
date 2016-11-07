import tensorflow as tf

from utility.Model_Parameter import *
from utility.Tools import IMAGE_INPUT_WIDTH
from utility.Tools import IMAGE_INPUT_HEIGH
##
#L1 inlucde 2 models,  predict the postion of  two eyes centre by
#using the whole image of face and the half whole image.
input_channel = 1
kernel_maps_1= 32
kernel_maps_2= 64
kernel_maps_3= 128
kernel_maps_4= 256
kernel_maps_5= 512

# 0 :pool,parameter of weights,after pool

fc_para_1 = 512
fc_para_2 = 512
label_szie = 4



def build_deep_cnn(input_datas,kernels,is_training):
    weight  = weight_variable(kernels)
    bias    = bias_variable(kernels[3])
    conv_ = tf.nn.bias_add(conv(input_datas, weight), bias)
    bn_conv = norm_layer(conv_, is_training)
    pool = max_pool_2x2(tf.nn.relu(bn_conv))
    return pool

def build_deep_fc(input_datas,shape,is_training,keep_prob):
        weight = weight_variable(shape)
        bias = bias_variable(shape[1])
        z = tf.matmul(input_datas, weight) + bias
        bn_fc= norm_layer(z, is_training)
        fc = tf.nn.dropout(tf.nn.relu(bn_fc), keep_prob)
        return fc


def L1_model_with_bn(input_datas, keep_prob,is_training):
    #reshape if nedd
    kernels = [[3, 3, input_channel, kernel_maps_1],
               [3, 3, kernel_maps_1, kernel_maps_2],
               [3, 3, kernel_maps_2, kernel_maps_3],
               [3, 3, kernel_maps_3, kernel_maps_4],
               [3, 3, kernel_maps_4, kernel_maps_5]]
    number_kernels= kernels[0].shape
    input_layer = input_datas
    for i in number_kernels:
        input_layer = build_deep_cnn(input_layer,is_training)

    #full connect
    fc_para_0 = IMAGE_INPUT_HEIGH //number_kernels * IMAGE_INPUT_WIDTH // number_kernels  * kernel_maps_5
    fc_weigths = [[fc_para_0,fc_para_1],[fc_para_1,fc_para_2]]
    input_fc = tf.reshape(input_layer,[-1, fc_weigths[0]])
    for i in fc_weigths[0].shape:
        input_fc = build_deep_fc(input_fc,fc_weigths,is_training,keep_prob)

    #last
    fc_w = weight_variable([512, 4])
    fc_b = bias_variable([4])
    labels = tf.matmul(input_fc, fc_w) + fc_b

    return labels


