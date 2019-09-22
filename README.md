# Insight Project: 3D Object Reconstruction
This work is for the Insight Project. Many deep learning algorithms use 2.5D images, which including the depth map, to predict 3D. This project uses multiple images with different views to reconstruct 3D image instead of using the depth map. For this, 3D-R2N2 ([paper](http://arxiv.org/abs/1604.00449)) has been used. 
Since the official source code ([link](https://github.com/chrischoy/3D-R2N2.git)) is implemented Theano, the Tensorflow implementation in [here](https://github.com/micmelesse/3D-reconstruction-with-Neural-Networks.git) has been used for this project. 

# Week 2
- [ ] Implement SE-ResNet (do a double check)
- [ ] Consider to use ShapeNetv2 data (now I'm using ShapeNetv1) 
- [ ] Make a test script to run the model with test images
- [ ] Make a more efficient script to obtain mIoU
- [ ] Add DeepLab inference code in `lib` 

# Week 1
- [x] Need to add a script to pull data from S3

# How to run
Followed the framework built in [here](https://github.com/micmelesse/3D-reconstruction-with-Neural-Networks.git). For now, we need to run with the `debug_network` branch. 
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
Training parameters are stored in `configs/params.json`. Please check the number of dataset first. Download and preprocess ShapeNet data (chair only) from S3
```
sh build/preprocess_dataset.sh
```
To run the training,
```
python run.py
```
