#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class Voxel_Softmax:
    def __init__(self, Y, logits):
        with tf.name_scope("Loss_Voxel_Softmax"):
            label = Y
            epsilon = 1e-10
            self.softmax = tf.clip_by_value(
                tf.nn.softmax(logits), epsilon, 1-epsilon)
            log_softmax = tf.log(self.softmax)
            # log_softmax = tf.nn.log_softmax(self.logits)  # avoids log(0)
            # label = tf.one_hot(Y, 2) # one hot encoding is done in preprocessing
            cross_entropy = tf.reduce_sum(-tf.multiply(label,
                                                       log_softmax), axis=-1)
            losses = tf.reduce_mean(cross_entropy, axis=[1, 2, 3])
            self.loss = tf.reduce_mean(losses)


class Focal_Loss:
    """
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    """
    def __init__(self, Y, logits, alpha=0.90, gamma=0.5):
        with tf.name_scope("Focal_Loss"):
            label = Y
            epsilon = 1e-10
            self.pred = tf.clip_by_value(
                tf.nn.sigmoid(logits), epsilon, 1-epsilon)
            ## cross-entropy
            #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)
            #self.pred = tf.clip_by_value(
            #    tf.nn.softmax(logits), epsilon, 1-epsilon)
            log_pred = tf.log(self.pred)
            p_t = tf.reduce_sum(-tf.multiply(label,self.pred), axis=-1)
            cross_entropy = tf.reduce_sum(-tf.multiply(label,log_pred), axis=-1)

            #alpha_ = label * alpha * (1.-label) * (1.-alpha)
            _alpha = label[...,1] * alpha + label[...,0] * (1.-alpha)

            losses = tf.multiply(tf.pow( _alpha*(1.-p_t), gamma), cross_entropy)
            losses = tf.reduce_mean(losses, axis=[1, 2, 3])

            self.loss = tf.reduce_mean(losses)


