# Installation
This tool need to be installed before use. All the requirements are in `requirements.txt`. Please install pytorch and torchvision dependencies. You can install this tool on a GPU machine using:

```
git clone https://github.com/kew6688/MicroNuclei_Detection.git && cd MicroNuclei_Detection

pip install -e .
```
 
 # Dataset
- NucRec
- DeepCell
- Kate

# Requirements
NucRec: 
    pytorch
    python 3.11

deepcell:
    module load StdEnv/2020
    avail_wheels tensorflow --all_versions --all_python | grep 2.8.0
    module load python/3.10

# Classification
Pretrained ResNet101 running on NucRec
Crop specific region to generate data from Kate

# Segmentation
DeepCell
Vision transformer
YOLOv10

# Tracking

# Generate
Multi task diffusion

# Usage:

## Project Structure:

```bash
├── base_ml               # Basic Machine Learning Code: CLI, Trainer, Experiment, ...
├── mn_classification     # Cell classification training and inference files
│   ├── utils             # Utils code (generate dataset, generate tiles contained mn)
│   ├── data_load.py      # Datasets loader (PyTorch)
│   ├── test.py           # Inference code for experiment statistics and plots
│   ├── train.py          # Trainer functions to train networks
│   └── main.py           # Run file to start an experiment
├── mn_segmentation       # Cell Segmentation training and inference files
│   ├── datasets          # Datasets (PyTorch)
│   ├── experiments       # Specific Experiment Code for different experiments
│   ├── inference         # Inference code for experiment statistics and plots
│   ├── trainer           # Trainer functions to train networks
│   ├── utils             # Utils code
│   └── run_xxx.py        # Run file to start an experiment
├── models                # Machine Learning Models (PyTorch implementations)
│   ├── encoders          # Encoder networks (see ML structure below)
│   ├── pretrained        # Checkpoint of important pretrained models (needs to be downloaded from Google drive)
│   └── segmentation      # CellViT Code
```

# Dependency

https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder

build project from root dir. Import by prepending MN.

# Dataset 
The following is the overview of the dataset. The dataset can be download here, by downloading the dataset you agree that you have read and accept licence.

We save masks per image in a json file. It can be loaded as a dictionary in python in the below format:
```python
{
    "image"                 : str,               # Image filename
    "id"                    : int,               # Image id
    "annotation"            : annotation,
}

annotation {
    "segmentation"          : brush_info,        
    "poly"                  : [[x,y]],           # polygon coordinates around the objects in the mask
    "bbox"                  : [[x, y, w, h]],    # The box around the objects in the mask, in XYWH format
    "crop_box"              : [x, y, w, h],      # The crop of the image used to generate the mask, in XYWH format
}

brush_info {
    "format"                : str,               # Indicate the format of the mask, "rle" for this dataset
    "rle"                   : [x],               # Mask saved in COCO RLE format
    "original_width"        : int,               # Mask width for RLE encoding
    "original_height"       : int,               # Mask height for RLE encoding
```
The dataset directory:
```bash
Data/
├── images                # All the images
├── result.json           # Json file that stores all the annotations
├── masks                 # Masks that in numpy file for each image
└── labels                # Labels that in YOLO format for each image
```
