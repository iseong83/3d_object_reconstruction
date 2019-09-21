# Insight Project: 3D Object Reconstruction
This work is for the Insight Project. Many deep learning algorithms use 2.5D images, which including the depth map, to predict 3D. This project uses multiple images with different views to reconstruct 3D image instead of using the depth map. For this, 3D-R2N2 ([paper](http://arxiv.org/abs/1604.00449)) has been used. 
Since the official source code ([link](https://github.com/chrischoy/3D-R2N2.git)) is implemented Theano, the Tensorflow implementation in [here](https://github.com/micmelesse/3D-reconstruction-with-Neural-Networks.git) has been used for this project. 

# Week 1
- [x] Test the Tensorflow code
- [ ] Decide dataset: a single category only? or train with multiple categories
- [ ] Need to add a script to pull data from S3
- [ ] Make a test script to run the model with test images

# How to run
Following the framework [here](https://github.com/micmelesse/3D-reconstruction-with-Neural-Networks.git). For now, it needs to run in the `debug_network` branch. 
Change the branch
`git checkout debug_network`
Install the requiements
```
pip install -r requirements.txt
```
Create directories
```
sh build/setup_dir.sh
```
Download and preprocess ShapeNet data (chair only) from S3
```
sh build/preprocess_dataset.sh
```
To run the training,
```
python run.py
```
Training parameters are stored in `configs/params.json`
