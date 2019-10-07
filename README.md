# 3D Object Reconstruction
This is an open source package to generate 3D voxel of object from 2D images. The implementation is based on the [3D-R2N2][1] model with integration of the [SE block][2]. This package also leveraged an image segmentation tool ([DeepLab][4]) to apply the 3D reconstruction to real images. Since the official source code is implemented in Theano, the Tensorflow implementation in [this github repo][3] is used for this project. 
<!--
## Demo
This project result can be viewed in [here][0]. Note that this demo is running on CPU (m4.xlarge in AWS). 
-->
## Installation / Setup
Clone this repository to a machine with GPU.
```
git clone https://github.com/iseong83/3d_object_reconstruction.git
cd 3d_object_reconstruction
```
Create a virtual environment and install packages
```
conda create -n 3d_reco python=3.6 # create a virtual environment
source activate 3d_reco            # start the virual environment
python setup.py install            # install this package and requirements
```

## Run Inference and Demo Locally
To run the inference/demo with the trained weights, you can download it from [here][5] (trained with ShapeNet chair cateogory only). 
Unzip the downloaded file in the `models` directory. 
```
mkdir models
tar -xvf models.tar -C ./models
```
Note that, after unzip, the `freezed_model` directory should be in the `models` directory.   
To run the inference of the 3D object reconstruction on ShapeNet data
```
python tests/inference_3d.py
```
To run the inference on real images
```
python demo.py
```

## Train the model
### Setup directories and Download data
To setup directories (e.g. data), download ShapeNet data (`chair only`), and preprocess data; run
```
python scripts/setup_and_preprocess.py
```
Note that the traning parameters are stored in `configs/params.json`, which includes a setting of the number of data to preprocess (`DATASET_SIZE`).   
To start the training, run
```
python run.py
```

[0]: http://34.220.155.13:8501/
[1]: https://arxiv.org/abs/1604.00449
[2]: https://arxiv.org/abs/1709.01507
[3]: https://github.com/micmelesse/3D-reconstruction-with-Neural-Networks.git
[4]: https://arxiv.org/abs/1802.02611
[5]: https://drive.google.com/file/d/1huRGy5vbZUlWWeTGDWiXPAK_6Kbz3vAE/view?usp=sharing
