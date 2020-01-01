# Tensorflow 2.0 Realtime Multi-Person Pose Estimation

This repo contains a new upgraded version of the **keras_Realtime_Multi-Person_Pose_Estimation** project. It is ready for the new Tensorflow 2.0.

I added a new model based on MobileNetV2 for mobile devices.
You can train it from scratch in the same way as the CMU model. There is still room for performance improvement, like quantization training, which I will add as a next step.

[Download](https://www.dropbox.com/s/gif7s1qlie2xftd/best_pose_mobilenet_model.zip?dl=1) the model and checkpoints.

I didn't change much the augmentation process as the tensorpack does a good job. The only changes I have made are in fetching samples to the model. I added the interface Dataset as recommended by Tensorflow.

It is worth to mention that I purposely didn't use the Keras interface **model.compile, model.run** as I had problems with loss regularizers - I kept getting NaN after a few iterations. I suspect that the solution would be to add loss to the input tensor: *add_loss(tf.abs(tf.reduce_mean(x)))*. I will update the repo as soon as I get satisfactory results.

I added a visualization of final heatmaps and pafs in the Tensorboard.
Every 100 iterations, a single image is passed to the model. The predicted heatmaps and pafs are logged in the Tensorboard.
You can check this visual representation of prediction every few minutes/hours as it gives a good sense of how the training performs.

# Scripts and notebooks

This project contains the following scripts and jupyter notebooks:

**train_custom_loop.py** - training code for the CMU model. This is a new version of the training code from the old repo *keras_Realtime_Multi-Person_Pose_Estimation*. It has been upgraded to Tensorflow 2.0.

**train_custom_loop_mobilenet.py** - training code for smaller model. It is based on the MobilenetV2. Simplified model with just 2 stages.

**convert_to_tflite.py** - script used to create *TFLite* model based on checkpoint or keras h5 file.

**dataset_inspect.ipynb** - helper notebook to get more insights into what is generated from the dataset.

**test_pose_mobilenet.ipynb** - helper notebook to preview the predictions from the mobilenet-based model.

**test_pose_vgg.ipynb** - helper notebook to preview the predictions from the original vgg-based model.

**test_tflite_model.ipynb** - helper notebook to verify exported *TFLite* model.

# Installation

## Prerequisites

* download [dataset and annotations](http://cocodataset.org/#download) into a separate folder datasets, outside of this project:
```bash
    ├── datasets
    │   └── coco_2017_dataset
    │       ├── annotations
    │       │   ├── person_keypoints_train2017.json
    │       │   └── person_keypoints_val2017.json
    │       ├── train2017/*
    │       └── val2017/*
    └── tensorflow_Realtime_Multi-Person_Pose_Estimation/*
```                
* install [CUDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda-downloads)

    If you use Anaconda, there is a simpler way. Just install the precompiled libs:
```bash    
    conda install -c anaconda cudatoolkit==10.0.130-0
```

## How to install (with tensorflow-gpu)


**Virtualenv**

```bash
pip install virtualenv
virtualenv venv
source venv/bin/activate

pip install -r requirements.txt
# ...or
pip install -r requirements_all.txt # completely frozen environment with all dependent libraries
```

**Anaconda**

```bash
conda create --name tf_pose_estimation_env
conda activate tf_pose_estimation_env

bash requirements_conda.txt
```
