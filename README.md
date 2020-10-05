# Tensorflow 2.0 Realtime Multi-Person Pose Estimation

## What's New

**Oct 5, 2020**
* Converted models to the new [string notation representation](https://github.com/michalfaber/tf_netbuilder)
* Added a new openpose singlenet model based on Mobilenet V3 [Single-Network Whole-Body Pose Estimation](https://arxiv.org/abs/1909.13423).
* Added dependency to the library [tf_netbuilder](https://github.com/michalfaber/tf_netbuilder)
* Old code is available under the tag: "v1.0"

#


This repo contains a new upgraded version of the **keras_Realtime_Multi-Person_Pose_Estimation** project plus some extra scripts and new models.


I added a visualization of final heatmaps and pafs in the Tensorboard.
Every 100 iterations, a single image is passed to the model. The predicted heatmaps and pafs are logged in the Tensorboard.
You can check this visual representation of prediction every few hours as it gives a good sense of how the training performs.

# Scripts and notebooks

This project contains the following scripts and jupyter notebooks:

**train_singlenet_mobilenetv3.py** - training code for the new model presented in this paper [Single-Network Whole-Body Pose Estimation](https://arxiv.org/abs/1909.13423). I replaced VGG with Mobilenet V3. Simplified model with just 3 pafs and 1 heatmap.

**train_2br_vgg.py** - training code for the old CMU model (2017). This is a new version of the training code from the old repo *keras_Realtime_Multi-Person_Pose_Estimation*. It has been upgraded to Tensorflow 2.0.

**convert_to_tflite.py** - conversion of trained models into *TFLite*.

**demo_image.py** - pose estimation on the provided image.

**demo_video.py** - pose estimation on the provided video.

**inspect_dataset.ipynb** - helper notebook to get more insights into what is generated from the datasets.

**test_openpose_singlenet_model.ipynb** - helper notebook to preview the predictions from the singlenet model.

**test_openpose_2br_vgg_model.ipynb** - helper notebook to preview the predictions from the original vgg-based model.

**test_tflite_models.ipynb** - helper notebook to verify exported *TFLite* model.
  

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


## Install

**Virtualenv**

```bash
pip install virtualenv
virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Examples
```bash
python convert_to_tflite.py --weights=[path to saved weights] --tflite-path=openpose_singlenet.tflite --create-model-fn=create_openpose_singlenet
python demo_image.py --image=resources/ski_224.jpg --output-image=out1.png --create-model-fn=create_openpose_singlenet
python demo_image.py --image=resources/ski_368.jpg --output-image=out2.png --create-model-fn=create_openpose_2branches_vgg
python demo_video.py --video=resources/sample1.mp4 --output-video=sample1_out1.mp4 --create-model-fn=create_openpose_2branches_vgg --input-size=368 --output-resize-factor=8 --paf-idx=10 --heatmap-idx=11
python demo_video.py --video=resources/sample1.mp4 --output-video=sample1_out2.mp4 --create-model-fn=create_openpose_singlenet --input-size=224 --output-resize-factor=8 --paf-idx=2 --heatmap-idx=3
```
