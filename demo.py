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
            type=str, default='./models/freezed_model/epoch_299')
    parser.add_argument("--seg_model", help="The model name for image semenation", 
            type=str, default='xception_coco')
    parser.add_argument("--data", help="Example data path", 
            type=str, default='./examples/chair_k')
    args = parser.parse_args()
    return args

def resize_images(resized_im, seg_map):
    size = resized_im.size
    kk = np.nonzero(seg_map)
    xmin, xmax = min(kk[0]), max(kk[0])
    ymin, ymax = min(kk[1]), max(kk[1])
       
    seg_map = seg_map[xmin:xmax,ymin:ymax]
    resized_im = np.array(resized_im)[xmin:xmax,ymin:ymax]
    resized_im = Image.fromarray(resized_im)
    
    return resized_im, seg_map

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
            new_img, seg_map = model.run(img)
            # resize img
            resized_im, seg_map = resize_images(new_img, seg_map)
            mask = np.repeat(seg_map[:,:,np.newaxis],3,axis=2)
            object = np.where(mask>0, np.array(resized_im), 0)
            # resize them

            max_size = max(resized_im.size)
            ratio = 120./max_size # to make a room
            #ratio = d_size/max_size # to make a room
            size = tuple([int(x*ratio) for x in resized_im.size])
            object = Image.fromarray(object)
            resized_im.thumbnail(size, Image.ANTIALIAS)
            object.thumbnail(size, Image.ANTIALIAS)
            new_im = Image.new("RGB",(d_size,d_size))
            new_im.paste(object,((d_size-size[0])//2,(d_size-size[1])//2))
            print ('-->', np.shape(new_im), np.shape(resized_im))
            original_img.append(new_img)
            segmented_img.append(new_im)
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
    if org_img.shape[0] < 3:
        vis.multichannel(org_img[0])
        vis.multichannel(seq_img[0])
    else:
        vis.img_sequence(seg_img)
        vis.img_sequence(org_img)

    seg_img = preprocessor.Preprocessor_npy(np.expand_dims(seg_img,axis=0)).out_tensor

    # make inference
    t1 = time.time()
    out = net.predict(seg_img[:,:,:,:,0:3])
    t2 = time.time()
    out = out[0]
    bg = out[:,:,:,0]
    fg = out[:,:,:,1]
    #fg[fg<=0.2] = 0
    out = np.stack([bg,fg],axis=-1)
    
    print ("Inference time {} sec".format(t2-t1))
    # show inference
    vis.voxel_binary(out)
    plt.show()

if __name__ == '__main__':
    main()
