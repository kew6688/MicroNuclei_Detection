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
      "area": [x,...],        # list of mask area
      "bbox": [...],         # list of bounding box, 
                        for mn bbox: (xmin, ymin, xmax, ymax)
                        for nuc bbox: (x, y, w, h) 
      "parent": [id,...]       # assigned parent nuclei, mn only
}

```
# Experiements
image process workflow: add multi-processing
