import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Reco3D.lib.dataset as dataset
import Reco3D.lib.network as network
from Reco3D.lib import preprocessor
import Reco3D.lib.vis as vis
from Reco3D.lib import metrics
import argparse
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="The model path", 
            type=str, default='./models_local/model_2019-09-15_22:25:37/epoch_153')
    parser.add_argument("--data", help="Example data path", 
            type=str, default='./examples/chair_a')
    parser.add_argument("--rnd", help="If want to make with random dataset", action='store_true', default=False)

 
    args = parser.parse_args()
    return args

def load_images(img_path, n_views=5):
    # load example images
    print ('Loading Images at {}'.format(img_path))
    filenames = list(set(os.path.join(img_path,n) for n in os.listdir(img_path) if n.endswith(".png")))
    # let's shuffle the images
    img_data = dataset.load_imgs(filenames)
    print ("Loaded example")
    return img_data

def main():
    args = get_args()
    model_dir = args.path.strip('/')
    image_dir = args.data
    random_data = args.rnd

    print ('Loading the model {}'.format(model_dir))
    net=network.Network_restored(model_dir)
    print ('Loaded the model')

    if not random_data:
        X = load_images(image_dir)
    else:
        X, Y = dataset.load_random_sample()
    # show example image
    vis.multichannel(X[0])
    X = preprocessor.Preprocessor_npy(np.expand_dims(X,axis=0)).out_tensor
    print (X.shape)

    # make inference
    t1 = time.time()
    #out = net.predict(X[:,:,:,0:3])
    out = net.predict(X[:,0,:,:,0:3])
    t2 = time.time()
    
    print ("Inference time {} sec".format(t2-t1))
    # show inference
    vis.voxel_binary(out[0])
    plt.show()

if __name__ == '__main__':
    main()
