import tensorflow as tf
import numpy as np
from Reco3D.lib import utils


def shuffle_sequence(value):
    with tf.name_scope("shuffle_sequence"):
        ret = tf.transpose(value, [1, 0, 2, 3, 4])
        ret = tf.random_shuffle(ret)
        ret = tf.transpose(ret, [1, 0, 2, 3, 4])
    return value


class Preprocessor():
    def __init__(self, X):
        with tf.name_scope("Preprocessor"):
            params = utils.read_params()
            if params["TRAIN"]["TIME_STEP_COUNT"] == "RANDOM":
                n_timesteps = tf.random_uniform(
                    [], minval=1, maxval=25, dtype=tf.int32)
                tf.summary.scalar("n_timesteps", n_timesteps)
            elif isinstance(params["TRAIN"]["TIME_STEP_COUNT"], int) and params["TRAIN"]["TIME_STEP_COUNT"] > 0:
                n_timesteps = params["TRAIN"]["TIME_STEP_COUNT"]
            else:
                n_timesteps = tf.shape(X)[1]

            n_batchsize = tf.shape(X)[0]
            X_dropped_alpha = X[:, :, :, :, 0:3]  # drop alpha channel
            X_cropped = tf.random_crop(
                X_dropped_alpha, [n_batchsize, n_timesteps, 127, 127, 3])   # randomly crop

            if params["TRAIN"]["SHUFFLE_IMAGE_SEQUENCE"]:
                X_shuffled = shuffle_sequence(X_cropped)
                self.out_tensor = X_shuffled
            else:
                self.out_tensor = X_cropped

class Preprocessor_npy():
    def __init__(self, X):
        params = utils.read_params()
        if params["TRAIN"]["TIME_STEP_COUNT"] == "RANDOM":
            n_timesteps = np.random.randint(0, 25)
        elif isinstance(params["TRAIN"]["TIME_STEP_COUNT"], int) and params["TRAIN"]["TIME_STEP_COUNT"] > 0:
            n_timesteps = params["TRAIN"]["TIME_STEP_COUNT"]
        else:
            n_timesteps = np.shape(X)[1]

        n_batchsize = np.shape(X)[0]
        X_dropped_alpha = X[:, :, :, :, 0:3]  # drop alpha channel
        if np.shape(X)[1] < n_timesteps:
            n_timesteps = np.shape(X)[1]
        X_cropped = self._crop_image(X_dropped_alpha, [n_batchsize, n_timesteps, 127, 127, 3])
        self.out_tensor = X_cropped
    def _crop_image(self, X, sizes):
        views = np.random.randint(0,X.shape[1],sizes[1])
        height, width = X.shape[2], X.shape[3]
        rangeh = (height-sizes[2])//2 if height>sizes[2] else 0
        rangew = (width-sizes[3])//2 if width>sizes[3] else 0
        assert rangeh>0
        assert rangew>0
        cropped = X[:,views,rangeh:rangeh+sizes[2],rangew:rangew+sizes[3],:]
        return cropped



