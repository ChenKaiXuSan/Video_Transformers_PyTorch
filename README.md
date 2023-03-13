<div align="center">    
 
# Video Transformers for Adult Spinal Deformity classification

</div>
 
## Description   

📓 This project made with the PyTorch, PyTorch Lightning, PyTorch Video and Hugging Face.

This project implements the task of classifying different adult spinal deformity diseases.

The current phase performs a binary classification task for four different disease. classification of ASD and non-ASD.

In this project, dataloader used from PytorchVideo, some model from PytorchVideo (3D CNN structure) and Hugging Face (video transformers).
Pytorch Lightning provides the overall framework of the program and PyTorch is the basic underlying framework.

Detailed comments are written for most of the methods and classes.
Have a nice code. 😄

## Folder structure  
todo

## How to run

1. install dependencies

``` bash
# clone project   
git clone https://github.com/ChenKaiXuSan/Walk_Video_PyTorch.git

# install project   
cd Walk_Video_PyTorch/ 
pip install -e .   
pip install -r requirements.txt
```

2. navigate to any file and run it.

```bash
# module folder
cd Walk_Video_PyTorch/

# run module 
python project/main.py [option] > logs/output_log/xxx.log 
```

# Experimental setup
todo 

## Dataset
todo 

## About the lib  

Stop building wheels 😄

### PyTorch Lightning  

[PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) is the deep learning framework for professional AI researchers and machine learning engineers who need maximal flexibility without sacrificing performance at scale. Lightning evolves with you as your projects go from idea to paper/production.

### PyTorch Video  

[link](https://pytorchvideo.org/)
A deep learning library for video understanding research.

### Hugging Face 

[Hugging Face](https://huggingface.co/transformers) 🤗 Transformers provides thousands of pretrained models to perform tasks on different modalities such as text, vision, and audio.

This project mainly uses video transformers pre-trained model and to fine-tune it.

### detectron2

[Detectron2](https://detectron2.readthedocs.io/en/latest/index.html) is Facebook AI Research's next generation library that provides state-of-the-art detection and segmentation algorithms. It is the successor of Detectron and maskrcnn-benchmark. It supports a number of computer vision research projects and production applications in Facebook.

### Torch Metrics

[TorchMetrics](https://torchmetrics.readthedocs.io/en/latest/) is a collection of 80+ PyTorch metrics implementations and an easy-to-use API to create custom metrics.

