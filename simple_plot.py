import glob
import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import lib.dataset as dataset
import lib.network as network
import lib.utils as utils
import lib.vis as vis
from PIL import Image
from collections import deque
from tensorflow.python.tools import inspect_checkpoint
from moviepy.editor import *
params=utils.read_params()
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# models_remote=utils.list_folders(params["DIRS"]["MODELS_REMOTE"])
models_remote=utils.list_folders(params["DIRS"]["MODELS_LOCAL"])
window_width=1

raw_experiment_params=pd.DataFrame()
raw_experiment_data=pd.DataFrame()
for i,m in enumerate(models_remote):
    
    model_params=utils.get_model_params(m)
    val_loss=utils.get_latest_loss(m,"val")


#     if model_params["TRAIN"]["EPOCH_COUNT"]-1==utils.get_latest_epoch_index(m):
    raw_experiment_params=raw_experiment_params.append(model_params["TRAIN"],ignore_index=True) 
    raw_experiment_data=raw_experiment_data.append(pd.Series([val_loss.mean()]),ignore_index=True)
        
    
raw_experiment_data=raw_experiment_data.round(3)

utils.fix_nparray(utils.get_latest_epoch(models_remote[-1])+"/train_loss.npy")

model_dir=params["SESSIONS"]["LONGEST"]

net=network.Network_restored(model_dir)

x,y=dataset.load_random_sample()
print (x.shape[0])
plt.plot(x[0])
plt.show()
