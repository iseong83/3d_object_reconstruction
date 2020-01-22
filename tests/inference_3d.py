import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Reco3D.lib.dataset as dataset
import Reco3D.lib.network as network
from Reco3D.lib import preprocessor
import Reco3D.lib.vis as vis
from Reco3D.lib import metrics
from PIL import Image
import argparse
import time
import math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="The model path", 
            type=str, default='./models/freezed_model/epoch_299')
    parser.add_argument("--data", help="Example data path", 
            type=str, default='./examples/chair_g')
    parser.add_argument("--rnd", help="use ShapeNet dataset", action='store_true', default=False)
    parser.add_argument("--test", help="Testing with Pix3D", action='store_true', default=False)

 
    args = parser.parse_args()
    return args

def load_images(img_path, n_views=1):
    # load example images
    print ('Loading Images at {}'.format(img_path))
    filenames = list(set(os.path.join(img_path,n) for n in os.listdir(img_path) if n.endswith(".png") or n.endswith('.jpg')))
    img_data = dataset.load_imgs(filenames)

    # resize if the input images larger than 137
    min_size, max_size = min(np.shape(img_data[0])[:2]), max(np.shape(img_data[0])[:2])
    if min_size > 137:
        ret = []
        for i in range(np.shape(img_data)[0]):
            img = Image.fromarray(img_data[i])
            size = math.ceil(max_size*137/min_size)
            img.thumbnail((size,size), Image.ANTIALIAS)
            ret.append(img)
        img_data = np.stack(ret)

    print ("Loaded example")
    return img_data

def main():
    args = get_args()
    model_dir = args.path.strip('/')
    image_dir = args.data
    random_data = args.rnd
    test = args.test

    nviews = 5

    print ('Loading the model {}'.format(model_dir))
    net=network.Network_restored(model_dir)
    print ('Loaded the model')

    if random_data:
        X, Y = dataset.load_random_sample()
    elif test:
        X, Y = dataset.load_random_data_Pix3D()
    else:
        X = load_images(image_dir, n_views=nviews)

    # show example image
    print ('---->',X.shape)
    if len(np.shape(X)) < 4:
        vis.multichannel(X)
    else:
        vis.multichannel(X[0])
    X = preprocessor.Preprocessor_npy(np.expand_dims(X,axis=0)).out_tensor
    print (X.shape)

    # make inference
    t1 = time.time()
    out = net.predict(X[:,:nviews,:,:,0:3])
    t2 = time.time()
    
    print ("Inference time {} sec".format(t2-t1))
    # show inference
    if test or random_data:
        vis.voxel_binary(Y)
    print (np.shape(out))
    out = out[0]
    bg = out[:,:,:0]
    fg = out[:,:,:1]
    fg[fg<0.3] = 0
    out = np.stack([bg,fg],axis=-1)
    vis.voxel_binary(out[0])
    plt.show()

if __name__ == '__main__':
    main()
