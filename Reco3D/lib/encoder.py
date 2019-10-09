import tensorflow as tf
tf.disable_v2_behavior()
import numpy as np
from Reco3D.lib import utils
from numpy.random import choice

# to map each view into layers
def map_images(fcn, sequence, name):
    with tf.name_scope(name):
        out = list()
        for i in range(sequence.get_shape()[1]):
            out.append(fcn(sequence[:,i,...]))
        ret = tf.stack(out,axis=1)
    return ret

# fully connected layer using the dense function and no bias
def fc_sequence(sequence, units):
    with tf.name_scope('fc_sequence'):
        def dense(x):
            return tf.layers.dense(inputs=x, use_bias=False, units=units)
        ret = tf.map_fn(dense, sequence, name='fc_map')
        #ret = map_images(dense, sequence, name='fc_map')
    return ret

# global averaging pooling
def global_average_pooling(sequence):
    with tf.name_scope('global_average_pooling'):
        def global_avg_pool(x):
            return tf.reduce_mean(x, axis=[1,2]) # [1,2] = [H, W] of image
        ret = tf.map_fn(global_avg_pool, sequence, name='global_avg_poo_map')
        #ret = map_images(global_avg_pool, sequence, name='global_avg_pool')
    return ret

# sigmoid function 
def sigmoid_sequence(sequence):
    with tf.name_scope('sigmoid_sequence'):
        def sigmoid(x):
            return tf.nn.sigmoid(x)
        ret = tf.map_fn(sigmoid, sequence, name='sigmoid_map')
        #ret = map_images(sigmoid, sequence, name='sigmod_map')
    return ret

# batch normalization
def batch_normalization(sequence, out_featuremap_count, training):
    with tf.name_scope('batch_normalization'):
        beta = tf.Variable(tf.constant(0.0, shape=[out_featuremap_count]),
                                     name='beta', trainable=training)
        gamma = tf.Variable(tf.constant(1.0, shape=[out_featuremap_count]),
                                      name='gamma', trainable=training)
        ema = tf.train.ExponentialMovingAverage(decay=0.9)
        def batch_norm(x):
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
            mean, var = tf.cond(tf.cast(training, tf.bool),
                                mean_var_with_update,
                                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
            return normed
        ret = tf.map_fn(batch_norm, sequence, name='batch_norm_map')
        #ret = map_images(batch_norm, sequence, name='batch_norm')
    return ret


def conv_sequence(sequence, in_featuremap_count, out_featuremap_count, initializer=None, K=3, S=[1, 1, 1, 1], D=[1, 1, 1, 1], P="SAME"):
    with tf.name_scope("conv_sequence"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        kernel = tf.Variable(
            init([K, K, in_featuremap_count, out_featuremap_count]), name="kernel")
        bias = tf.Variable(init([out_featuremap_count]), name="bias")

        def conv2d(x): return tf.nn.bias_add(tf.nn.conv2d(
            x, kernel, S, padding=P, dilations=D, name="conv2d"), bias)
        ret = tf.map_fn(conv2d, sequence, name="conv2d_map")
        #ret = map_images(conv2d, sequence, name='conv2d_map')

        tf.add_to_collection("feature_maps", ret)

        # visualization code
        params = utils.read_params()
        image_count = params["VIS"]["IMAGE_COUNT"]
        if params["VIS"]["KERNELS"]:
            kern_1 = tf.concat(tf.unstack(kernel, axis=-1), axis=-1)
            kern_2 = tf.transpose(kern_1, [2, 0, 1])
            kern_3 = tf.expand_dims(kern_2, -1)
            tf.summary.image("2d kernel", kern_3, max_outputs=image_count)

        if params["VIS"]["FEATURE_MAPS"]:
            feature_map_1 = tf.concat(tf.unstack(ret, axis=4), axis=2)
            feature_map_2 = tf.concat(
                tf.unstack(feature_map_1, axis=1), axis=2)
            feature_map_3 = tf.expand_dims(feature_map_2, -1)
            tf.summary.image("feature_map", feature_map_3,
                             max_outputs=image_count)

        if params["VIS"]["HISTOGRAMS"]:
            tf.summary.histogram("kernel", kernel)
            tf.summary.histogram("bias", bias)

        if params["VIS"]["SHAPES"]:
            print(ret.shape)
    return ret


def max_pool_sequence(sequence, K=[1, 2, 2, 1], S=[1, 2, 2, 1], P="SAME"):
    with tf.name_scope("max_pool_sequence"):
        def max_pool(a): 
            return tf.nn.max_pool(a, K, S, padding=P)
        ret = tf.map_fn(max_pool, sequence, name="max_pool_map")
        #ret = map_images(max_pool, sequence, name='max_pool_map')
    return ret


def relu_sequence(sequence):
    with tf.name_scope("relu_sequence"):
        ret = tf.map_fn(tf.nn.relu, sequence, name="relu_map")
        #ret = map_images(tf.nn.relu, sequence, name='relu_map')
    return ret

def fully_connected_sequence(sequence, in_units=1024, out_units=1024, initializer=None):
    with tf.name_scope("fully_connected_sequence"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        weights = tf.Variable(
            init([in_units, out_units]), name="weights")
        bias = tf.Variable(init([out_units]), name="bias")
        def forward_pass(a): return tf.nn.bias_add(
            tf.matmul(a, weights), bias)

        ret = tf.map_fn(forward_pass, sequence, name='fully_connected_map')
        #ret = map_images(forward_pass, sequence, name='fully_connected_map')

        params = utils.read_params()
        if params["VIS"]["HISTOGRAMS"]:
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("bias", bias)
        if params["VIS"]["SHAPES"]:
            print(ret.shape)
 
    return ret


def flatten_sequence(sequence):
    with tf.name_scope("flatten_sequence"):
        ret = tf.map_fn(
            tf.contrib.layers.flatten,  sequence, name="flatten_map")
        #ret = map_images(tf.contrib.layers.flatten, sequence, name='flatten_map')
    return ret

# transition layer before applying squeeze and excitation layers
def transition_layer(sequence, out_featuremap_count, init):
    with tf.name_scope('transition_layer'):
        conv1 = conv_sequence(sequence, out_featuremap_count, out_featuremap_count, K=1, initializer=init)
        ret = batch_normalization(conv1, out_featuremap_count, training=True)
    return ret


def block_seresnet_encoder(sequence, out_featuremap_count, ratio=4, initializer=None, pool=True):
    # sequeeze excitation encoder (SENet)
    with tf.name_scope('block_seresnet_encoder') :
        out = sequence
        out = transition_layer(out, out_featuremap_count, init=initializer)
        squeeze = global_average_pooling(out)

        excitation = fully_connected_sequence(squeeze, in_units=out_featuremap_count, out_units=out_featuremap_count//ratio)
        excitation = relu_sequence(excitation)
        excitation = fully_connected_sequence(excitation, in_units=out_featuremap_count//ratio, out_units=out_featuremap_count)
        excitation = sigmoid_sequence(excitation)
        #excitation = tf.reshape(excitation, [-1,out.get_shape()[1],1,1,out_featuremap_count])
        excitation = tf.reshape(excitation, [tf.shape(out)[0],-1,1,1,out_featuremap_count])
        scale = out * excitation
        ret = relu_sequence(sequence+scale)
    return ret


def block_simple_encoder(sequence, in_featuremap_count, out_featuremap_count,  K=3, D=[1, 1, 1, 1], initializer=None):
    with tf.name_scope("block_simple_encoder"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        conv = conv_sequence(sequence, in_featuremap_count,
                             out_featuremap_count, K=K, D=D, initializer=init)
        pool = max_pool_sequence(conv)
        relu = relu_sequence(pool)
    return relu


def block_residual_encoder(sequence, in_featuremap_count, out_featuremap_count,  K_1=3, K_2=3, K_3=1, D=[1, 1, 1, 1], initializer=None, pool=True):
    with tf.name_scope("block_residual_encoder"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        out = sequence
        if K_1 != 0:
            conv1 = conv_sequence(out, in_featuremap_count,
                                  out_featuremap_count, K=K_1, D=D, initializer=init)
            relu1 = relu_sequence(conv1)
            out = relu1

        if K_2 != 0:
            conv2 = conv_sequence(out, out_featuremap_count,
                                  out_featuremap_count, K=K_2, D=D, initializer=init)
            relu2 = relu_sequence(conv2)
            out = relu2

        if K_3 != 0:
            # Fixed: the original code applied the skip connection only after the last conv layer
            conv3 = conv_sequence(sequence, in_featuremap_count,
                                  out_featuremap_count, K=K_3, D=D, initializer=init)
            out = conv3 + relu2

        if pool:
            pool = max_pool_sequence(out)
            out = pool

        return out


def block_dilated_encoder(sequence, in_featuremap_count, out_featuremap_count,  K_1=3, K_2=3, K_3=1, D=[1, 1, 1, 1], initializer=None, pool=True):
    with tf.name_scope("block_dilated_encoder"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        out = sequence
        if K_1 != 0:
            conv1 = conv_sequence(out, in_featuremap_count,
                                  out_featuremap_count, K=K_1, D=D, initializer=init)
            relu1 = relu_sequence(conv1)
            out = relu1

        if K_2 != 0:
            conv2 = conv_sequence(out, out_featuremap_count,
                                  out_featuremap_count, K=K_2, D=D, initializer=init)
            relu2 = relu_sequence(conv2)
            out = relu2

        if K_3 != 0:
            conv3 = conv_sequence(out, out_featuremap_count,
                                  out_featuremap_count, K=K_3, D=D, initializer=init)
            out = conv3 + relu2

        return out



# SEResNet
class SENet_Encoder:
    def __init__(self, sequence, feature_map_count=[96, 128, 256, 256, 256, 256], initializer=None):
        with tf.name_scope("SENet_Encoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_map_count)
            # convolution stack
            cur_tensor = block_residual_encoder(
                sequence, 3, feature_map_count[0], K_1=7, K_2=3, K_3=0, initializer=init, pool=False)
            cur_tensor = block_seresnet_encoder(
                cur_tensor, feature_map_count[0], initializer=init)
            # max pooling
            cur_tensor = max_pool_sequence(cur_tensor)


            for i in range(1, N):
                if i == 3:
                    # the original paper doesn't applied the skip connection in this step
                    cur_tensor = block_residual_encoder(
                            cur_tensor, feature_map_count[i-1], feature_map_count[i], K_3=0, initializer=init, pool=False)
                else:
                    cur_tensor = block_residual_encoder(
                            cur_tensor, feature_map_count[i-1], feature_map_count[i], initializer=init, pool=False)
                cur_tensor = block_seresnet_encoder(
                     cur_tensor, feature_map_count[i], initializer=init)
                # max pooling
                cur_tensor = max_pool_sequence(cur_tensor)

            # final block
            flat = flatten_sequence(cur_tensor, )
            fc0 = fully_connected_sequence(flat, in_units=1024)
            self.out_tensor = relu_sequence(fc0)


class Simple_Encoder:
    def __init__(self, sequence, feature_map_count=[96, 128, 256, 256, 256, 256], initializer=None):
        with tf.name_scope("Simple_Encoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_map_count)

            # convolution stack
            cur_tensor = block_simple_encoder(
                sequence, 3, feature_map_count[0], K=7, initializer=init)
            for i in range(1, N):
                cur_tensor = block_simple_encoder(
                    cur_tensor, feature_map_count[i-1], feature_map_count[i], initializer=init)

            # final block
            flat = flatten_sequence(cur_tensor)
            fc0 = fully_connected_sequence(flat)
            self.out_tensor = relu_sequence(fc0)


class Residual_Encoder:
    def __init__(self, sequence, feature_map_count=[96, 128, 256, 256, 256, 256], initializer=None):
        with tf.name_scope("Residual_Encoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            cur_tensor = block_residual_encoder(
                sequence, 3, feature_map_count[0], K_1=7, K_2=3, K_3=0, initializer=init)
            # convolution stack
            N = len(feature_map_count)
            for i in range(1, N):
                if i == 3:
                    # the original paper doesn't applied the skip connection in this step
                    cur_tensor = block_residual_encoder(
                            cur_tensor, feature_map_count[i-1], feature_map_count[i], K_3=0, initializer=init)
                else:
                    cur_tensor = block_residual_encoder(
                            cur_tensor, feature_map_count[i-1], feature_map_count[i], initializer=init)

            # final block
            flat = flatten_sequence(cur_tensor)
            fc0 = fully_connected_sequence(flat)
            self.out_tensor = relu_sequence(fc0)


class Dilated_Encoder:
    def __init__(self, sequence, feature_map_count=[96, 128, 256, 256, 256, 256], initializer=None):
        with tf.name_scope("Dilated_Encoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_map_count)

            # convolution stack
            cur_tensor = block_dilated_encoder(
                sequence, sequence.shape[-1], feature_map_count[0], D=[1, 2, 2, 1], initializer=init)
            for i in range(1, N):
                cur_tensor = block_dilated_encoder(
                    cur_tensor, feature_map_count[i-1], feature_map_count[i], D=[1, 2, 2, 1], initializer=init)

            # final block
            flat = flatten_sequence(cur_tensor)
            fc0 = fully_connected_sequence(flat)
            self.out_tensor = relu_sequence(fc0)


