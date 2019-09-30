import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Reco3D.lib.dataset as dataset
import Reco3D.lib.network as network
from Reco3D.lib import preprocessor
import Reco3D.lib.vis as vis
from Reco3D.lib.segmentation import *
from PIL import Image
import argparse
import time
import math

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reco_model", help="The model path", 
            type=str, default='./models_local/model_2019-09-15_22:25:37/epoch_153')
    parser.add_argument("--seg_model", help="The model name for image semenation", 
            type=str, default='mobile_coco')
    parser.add_argument("--data", help="Example data path", 
            type=str, default='./examples/chair_z')
    args = parser.parse_args()
    return args

def load_images(img_path, model, n_views=5):
    # load example images and apply segmentation
    print ('Loading Images at {}'.format(img_path))
    original_img = list()
    segmented_img = list()
    d_size = 137
    for root, _, files in os.walk(img_path):
        for i, file in enumerate(files):
            if i >= n_views: break
            img = load_image(os.path.join(root,file))
            # DeepLab for image segmentation
            resized_im, seg_map = model.run(img)
            mask = np.repeat(seg_map[:,:,np.newaxis],3,axis=2)
            object = np.where(mask>0, np.array(resized_im), 0)
            # resize them
            max_size = max(resized_im.size)
            ratio = d_size/max_size
            size = tuple([int(x*ratio) for x in resized_im.size])
            object = Image.fromarray(object)
            resized_im.thumbnail(size, Image.ANTIALIAS)
            object.thumbnail(size, Image.ANTIALIAS)
            new_im = Image.new("RGB",(d_size,d_size))
            new_im.paste(object,((d_size-size[0])//2,(d_size-size[1])//2))
            print ('-->', np.shape(new_im))
            original_img.append(resized_im)
            segmented_img.append(object)
    print ("Loaded example")
    return np.stack(original_img), np.stack(segmented_img)

def main():
    args = get_args()
    reco_model = args.reco_model.strip('/')
    seg_model  = args.seg_model
    image_dir  = args.data

    print ('Loading the models {} {}'.format(reco_model, seg_model))
    net = network.Network_restored(reco_model)
    seg = DeepLabModel(seg_model)
    print ('Loaded the model')
    
    org_img, seg_img = load_images(image_dir, seg)
    print (org_img.shape, seg_img.shape)

    # show example image
    #vis.multichannel(org_img[0])
    vis.img_sequence(org_img)
    vis.img_sequence(seg_img)
    seg_img = preprocessor.Preprocessor_npy(np.expand_dims(seg_img,axis=0)).out_tensor

    # make inference
    t1 = time.time()
    out = net.predict(seg_img[:,:,:,:,0:3])
    t2 = time.time()
    
    print ("Inference time {} sec".format(t2-t1))
    # show inference
    vis.voxel_binary(out[0])
    plt.show()

if __name__ == '__main__':
    main()
