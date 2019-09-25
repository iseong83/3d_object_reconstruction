import tensorflow as tf
from Reco3D.lib import utils

# batch normalization
def batch_normalization_vox(vox, out_featurevoxel_count, training):
    with tf.name_scope('batch_normalization'):
        beta = tf.Variable(tf.constant(0.0, shape=[out_featurevoxel_count]),
                                     name='beta', trainable=training)
        gamma = tf.Variable(tf.constant(1.0, shape=[out_featurevoxel_count]),
                                      name='gamma', trainable=training)
        ema = tf.train.ExponentialMovingAverage(decay=0.9)

        batch_mean, batch_var = tf.nn.moments(vox, [0,1,2,3], name='moments')
        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)
        mean, var = tf.cond(tf.cast(training, tf.bool),
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        ret = tf.nn.batch_normalization(vox, mean, var, beta, gamma, 1e-3)
    return ret

def conv_vox(vox, in_featurevoxel_count, out_featurevoxel_count, K=3, S=[1, 1, 1, 1, 1], D=[1, 1, 1, 1, 1], initializer=None, P="SAME"):
    # deconvolution
    with tf.name_scope("conv_vox"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        kernel = tf.Variable(
            init([K, K, K, in_featurevoxel_count, out_featurevoxel_count]), name="kernel")
        bias = tf.Variable(init([out_featurevoxel_count]), name="bias")
        ret = tf.nn.bias_add(tf.nn.conv3d(
            vox, kernel, S, padding=P, dilations=D, name="conv3d"), bias)
        tf.add_to_collection("feature_voxels", ret)

        # visualization code
        params = utils.read_params()
        image_count = params["VIS"]["IMAGE_COUNT"]
        if params["VIS"]["KERNELS"]:
            kern_1 = tf.concat(tf.unstack(kernel, axis=-1), axis=-1)
            kern_2 = tf.transpose(kern_1, [3, 0, 1, 2])
            kern_3 = tf.expand_dims(kern_2, -1)
            kern_4 = tf.concat(tf.unstack(kern_3, axis=1), axis=1)
            tf.summary.image("3d kernel", kern_4, max_outputs=image_count)

        if params["VIS"]["VOXEL_SLICES"]:
            vox_slice_1 = tf.unstack(ret, axis=4)[1]
            vox_slice_2 = tf.split(vox_slice_1, 4, axis=3)
            vox_slice_3 = tf.concat(vox_slice_2, axis=1)
            vox_slice_4 = tf.concat(tf.unstack(vox_slice_3, axis=-1), axis=2)
            vox_slice_5 = tf.expand_dims(vox_slice_4, -1)
            tf.summary.image("vox_slices", vox_slice_5,
                             max_outputs=image_count)

        if params["VIS"]["FEATURE_VOXELS"]:
            tf.summary.tensor_summary("feature_voxels", ret[0, :, :, :, 0])

        if params["VIS"]["HISTOGRAMS"]:
            tf.summary.histogram("kernel", kernel)
            tf.summary.histogram("bias", bias)

        if params["VIS"]["SHAPES"]:
            print(ret.shape)

    return ret


def unpool_vox(value):  # from tenorflow github board
    with tf.name_scope('unpool_vox'):
        sh = value.get_shape().as_list()
        dim = len(sh[1: -1])
        out = (tf.reshape(value, [-1] + sh[-dim:]))

        for i in range(dim, 0, -1):
            out = tf.concat([out, tf.zeros_like(out)], i)

        out_size = [-1] + [s * 2 for s in sh[1:-1]] + [sh[-1]]
        out = tf.reshape(out, out_size)

    return out


def relu_vox(vox):
    with tf.name_scope("relu_vox"):
        ret = tf.nn.relu(vox, name="relu")
    return ret

def sigmoid_vox(vox):
    with tf.name_scope("sigmoid_vox"):
        ret = tf.nn.sigmoid(vox, name='sigmoid')
    return ret

# global averaging pooling
def global_average_pooling(vox):
    with tf.name_scope('global_average_pooling'):
        ret = tf.reduce_mean(vox, axis=[1,2,3], name='global_avg_pool_vox') 
    return ret

def transition_vox(vox, out_featurevoxel_count, init):
    with tf.name_scope('transition_vox'):
        conv1 = conv_vox(vox, out_featurevoxel_count, out_featurevoxel_count, K=1, initializer=init)
        ret = batch_normalization_vox(conv1, out_featurevoxel_count, training=True)
    return ret

# fully connected layer using the dense function and no bias
def fc_vox(vox, units):
    with tf.name_scope('fc_vox'):
        ret = tf.layers.dense(inputs=vox, use_bias=False, units=units, name='fc')
    return ret

def fully_connected_vox(vox, in_units=1024, out_units=1024, initializer=None):
    with tf.name_scope("fully_connected_vox"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        weights = tf.Variable(
            init([in_units, out_units]), name="weights")
        bias = tf.Variable(init([out_units]), name="bias")
        def forward_pass(a): return tf.nn.bias_add(
            tf.matmul(a, weights), bias)
        ret = tf.nn.bias_add(tf.matmul(vox, weights), bias)

        params = utils.read_params()
        if params["VIS"]["HISTOGRAMS"]:
            tf.summary.histogram("weights", weights)
            tf.summary.histogram("bias", bias)
        if params["VIS"]["SHAPES"]:
            print(ret.shape)
 
    return ret


def block_seresnet_decoder(vox, out_featurevoxel_count, ratio=4, initializer=None, pool=True):
    # sequeeze excitation decoder (SENet)
    with tf.name_scope('block_seresnet_decoder') :
        out = vox
        out = transition_vox(out, out_featurevoxel_count, init=initializer)
        # squeeze layer
        squeeze = global_average_pooling(out)
        # excitation layers
        excitation = fully_connected_vox(squeeze, in_units=out_featurevoxel_count, out_units=out_featurevoxel_count//ratio)
        excitation = relu_vox(excitation)

        excitation = fully_connected_vox(excitation, in_units=out_featurevoxel_count//ratio, out_units=out_featurevoxel_count)
        excitation = sigmoid_vox(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,1,out_featurevoxel_count])
        scale = out * excitation
        return relu_vox(vox+scale)


def block_simple_decoder(vox, in_featurevoxel_count, out_featurevoxel_count, K=3, D=[1, 1, 1, 1, 1], initializer=None, unpool=False):
    with tf.name_scope("block_simple_decoder"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        conv = conv_vox(vox, in_featurevoxel_count, out_featurevoxel_count,
                        K=K,  D=D, initializer=init)
        if unpool:
            out = relu_vox(unpool_vox(conv))
        else:
            out = relu_vox(conv)

    return out


def block_residual_decoder(vox, in_featurevoxel_count, out_featurevoxel_count, K_1=3, K_2=3, K_3=1, D=[1, 1, 1, 1, 1], initializer=None, unpool=False):
    with tf.name_scope("block_residual_decoder"):
        if initializer is None:
            init = tf.contrib.layers.xavier_initializer()
        else:
            init = initializer

        out = vox
        if K_1 != 0:
            conv1 = conv_vox(out, in_featurevoxel_count,
                             out_featurevoxel_count, K=K_1, D=D, initializer=init)
            relu1 = relu_vox(conv1)
            out = relu1

        if K_2 != 0:
            conv2 = conv_vox(out, out_featurevoxel_count,
                             out_featurevoxel_count, K=K_2, D=D, initializer=init)
            relu2 = relu_vox(conv2)
            out = relu2

        if K_3 != 0:
            # Fixed: the original code applied the skip connection only after the last conv layer
            conv3 = conv_vox(vox, in_featurevoxel_count,
                             out_featurevoxel_count, K=K_3, D=D, initializer=init)
            out = conv3 + relu2

        if unpool:
            unpool = unpool_vox(out)
            out = unpool

    return out

class SENet_Decoder:
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("SENet_Decoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_vox_count)
            hidden_shape = hidden_state.get_shape().as_list()
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = block_residual_decoder(
                cur_tensor, hidden_shape[-1], feature_vox_count[0], initializer=init)
            cur_tensor = block_seresnet_decoder(
                cur_tensor, feature_vox_count[0], initializer=init)

            for i in range(1, N-1):
                unpool = True if i <= 2 else False
                cur_tensor = block_residual_decoder(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], initializer=init, unpool=False)
                cur_tensor = block_seresnet_decoder(
                    cur_tensor, feature_vox_count[i], initializer=init)
                if unpool: cur_tensor = unpool_vox(cur_tensor)
            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)



class Residual_Decoder:
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("Residual_Decoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_vox_count)
            hidden_shape = hidden_state.get_shape().as_list()
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = block_residual_decoder(
                cur_tensor, hidden_shape[-1], feature_vox_count[0], initializer=init)
            for i in range(1, N-1):
                unpool = True if i <= 2 else False
                cur_tensor = block_residual_decoder(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], initializer=init, unpool=unpool)

            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)


class Dilated_Decoder:
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("Dilated_Decoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_vox_count)
            hidden_shape = hidden_state.get_shape().as_list()
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = block_simple_decoder(
                cur_tensor, hidden_shape[-1], feature_vox_count[0], initializer=init)
            for i in range(1, N-1):
                unpool = True if i <= 2 else False
                cur_tensor = block_simple_decoder(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], D=[1, 2, 2, 2, 1], initializer=init, unpool=unpool)

            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)


class Simple_Decoder:
    def __init__(self, hidden_state, feature_vox_count=[128, 128, 128, 64, 32, 2], initializer=None):
        with tf.name_scope("Simple_Decoder"):
            if initializer is None:
                init = tf.contrib.layers.xavier_initializer()
            else:
                init = initializer

            N = len(feature_vox_count)
            hidden_shape = hidden_state.get_shape().as_list()
            cur_tensor = unpool_vox(hidden_state)
            cur_tensor = block_simple_decoder(
                cur_tensor, hidden_shape[-1], feature_vox_count[0], initializer=init)
            # the original paper implemented this part little differently. But, just keep the current network
            for i in range(1, N-1):
                unpool = True if i <= 2 else False
                cur_tensor = block_simple_decoder(
                    cur_tensor, feature_vox_count[i-1], feature_vox_count[i], initializer=init, unpool=unpool)

            self.out_tensor = conv_vox(
                cur_tensor, feature_vox_count[-2], feature_vox_count[-1], initializer=init)
