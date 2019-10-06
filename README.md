# Insight Project: 3D Object Reconstruction
This work is for the Insight Project. Many deep learning algorithms use 2.5D images, which including the depth map, to predict 3D. This project uses multiple images with different views to reconstruct 3D image instead of using the depth map. For this, 3D-R2N2 ([paper](http://arxiv.org/abs/1604.00449)) has been used. 
Since the official source code ([link](https://github.com/chrischoy/3D-R2N2.git)) is implemented Theano, the Tensorflow implementation in [here](https://github.com/micmelesse/3D-reconstruction-with-Neural-Networks.git) has been used for this project. 

# How to Train
Followed the framework built in [here](https://github.com/micmelesse/3D-reconstruction-with-Neural-Networks.git). 
To install the package
```
python setup.py install
```
Next step will create directories, download ShapeNet data (chair only), and preprocess data. Training parameters are stored in `configs/params.json` and `DATASET_SIZE` will determine the number of data to preprocess. 
```
python scripts/setup_and_preprocess.py
```
To run the training,
```
python run.py

```
