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

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

The pre-trained models can be download from huggingface:
- https://huggingface.co/kew1046/MaskRCNN-resnet50FPN
- https://huggingface.co/kew1046/MaskRCNN-swinFPN
  
After downloading the model, the usage of the end-to-end pipeline is described below.

## Usage:
Automated pipeline to process images. Further details will be add.
 
 This includes 
 - predict counts of micronuclei
 - predict masks
 - output a info dictionary with box, center location and size

Please refer to the examples in the [tutorial.ipynb](./notebooks/tutorial.ipynb) (open in [colab](https://colab.research.google.com/github/kew6688/MicroNuclei_Detection/blob/main/notebooks/tutorial.ipynb))

Compute scripts are provided.
```
# Run the main python script.
# The arguments should be
#       the folder for the input images (png, tif)
#       the final json file name
#       the process mode (ALL for both nuc and mn, NUC for only nuclei, MN for only micronuclei)
# Example:
#       >>> python image_process.py --src /home/test --dst test.json --mode ALL 
```

### Parameters and Arguments
| Parameter          | Short Form | Required | Default    | Type         | Description                                                                                       |
|--------------------|------------|----------|------------|--------------|---------------------------------------------------------------------------------------------------|
| `--src`            | `-s`       | Yes      | N/A        | String       | Pathway to image.                                                                                 |
| `--dst`            | `-d`       | Yes      | N/A        | String       | Pathway to output.                                                                                |
| `--mode`           | `-mod`     | Yes      | N/A        | String       | mode for output. Options: `["MN", "NUC", "ALL"]`                                                  |
| `--conf`           | `-c`       | No       | N/A        | Float        | confidence threshold for micronuclei detection, e.g. --conf 0.4                                   |
| `--out`            | `-o`       | No       | N/A        | Float        | Output format is contained mask (full) or only box (short), e.g. -o full/short                     |
| `--parent`         | `-p`       | No       | N/A        | Float        | Parent assign method, use closest center or edge to find nearest parent nuclei (edge by default)   |



## Project Structure:

```bash
├── checkpoints           # Download pretrained weight here
├── mn_classification     # Micronuclei classification training and inference files
│   ├── utils             # Utils code (generate dataset, generate tiles contained mn)
│   ├── data_load.py      # Datasets loader (PyTorch)
│   ├── test.py           # Inference code for experiment statistics and plots
│   ├── train.py          # Trainer functions to train networks
│   └── main.py           # Run file to start an experiment
├── mn_segmentation       # Micronuclei Segmentation training and inference files
│   ├── datasets          # Datasets (PyTorch)
│   ├── lib               # Application class for using the model
│   ├── models            # Networks architecture
│   ├── train             # Trainer functions to train networks
│   ├── tests             # Evaluation code
│   └── run.py            # Run file to start an experiment
└── notebooks             # Inference code for experiment statistics and plots
```

## Dependency

https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder

build project from root dir. Import by prepending MN.

## Dataset 
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
├── label_masks           # Masks that in numpy file for each image, instances are encoded as different colors (0 is background)
└── labels                # Labels that in YOLO format for each image
```

## Output format
Generate a json file with all the processed nuclei and mn information stored in the following format.

```
[image_info], size n == number of images:

[
    {
        "image": image_name,    # str
        "nuclei": nuc_info,
        "micronuclei": mn_info
    }
]

For each info:

{
      "coord": [[x1, y1],...],   `# list of center coordinates
      "area": [x,...],            # list of mask area
      "bbox": [...],              # list of bounding box, 
                                    for mn bbox: (xmin, ymin, xmax, ymax)
                                    for nuc bbox: (x, y, w, h) 
      "score": [x,...],           # list of prediction scores for each object
      "mask": [[...],...]         # list of rle encoding list for each object

      "parent": [id,...]          # assigned parent nuclei, mn only
}
```
Note: RLE encoding and decoding functions can be find in 
```
from mn_segmentation.lib.image_encode import mask2rle, rle_to_mask
rle = mask2rle(mask)
mask = rle_to_mask(rle,original_height,original_width)
```

## Updates
2025/3/10:
- Add output predictions' mask and confidence
- Add new parent assign method
- Add flag for output format and parent method
