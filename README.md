# Micronuclei detection

![sample](./sample_images/example1.png)


## Installation
This tool need to be installed before use. All the requirements are in `requirements.txt`. Please install pytorch and torchvision dependencies. 

You can install this tool on a GPU machine using:

```
git clone https://github.com/kew6688/MicroNuclei_Detection.git && cd MicroNuclei_Detection
pip install -e .
```

## Model Checkpoints
The pre-trained models can be download from huggingface by ```git clone```:
- https://huggingface.co/kew1046/MaskRCNN-resnet50FPN
- https://huggingface.co/kew1046/MaskRCNN-swinFPN
  
After downloading the model, the usage of the end-to-end pipeline is described below.

## Usage:
Automated pipeline to process images. Further details will be add [here](www...)
 
 This includes 
 - predict counts of micronuclei
 - predict masks
 - output a info dictionary with box, center location and size

Please refer to the examples in the [tutorial.ipynb](./notebooks/tutorial.ipynb) (open in [colab](https://colab.research.google.com/github/kew6688/MicroNuclei_Detection/blob/main/notebooks/tutorial.ipynb))

