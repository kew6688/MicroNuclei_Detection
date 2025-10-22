

<div align="center">
<img src="./sample_images/example1.png" alt="sample" width="600" height="500">
</div>

## 🏠 Introduction

The **MicroNuclei Detection** repository provides a comprehensive end-to-end pipeline for the automatic detection and quantification of micronuclei and their parent nuclei in microscopy images. Leveraging modern deep learning architectures combined with robust segmentation and classification workflows, this tool is designed to help researchers and practitioners in biomedical imaging quickly and accurately identify micronuclei, a key biomarker for cellular damage and genomic instability.

### Key Features

- A seamless integration of segmentation (for parent nuclei) and detection/classification (for micronuclei) modules — enabling you to feed in RGB images and receive annotated outputs with details such as bounding boxes, center coordinates, mask areas, and classification scores.

- Support for multiple modern back-ends and pretrained checkpoints (e.g., ResNet50, Swin, ResNet101) to enable rapid setup and experimentation. 

- A flexible command-line tool interface, enabling the processing of large image sets via a simple `python image_process.py --src ... --dst ...` invocation, with configurable modes (nuclei only, micronuclei only, or both), confidence thresholds, and parent-assignment strategies. 

- Support for segmentation-based datasets (via JSON annotations, RLE encoding, bounding boxes, and masks) and output formats clearly defined for downstream analytics and data pipelines. 

- Designed with modular code structure including data-loading, training, inference (for both segmentation and classification), and optional notebooks for statistical evaluation and plotting.

## 🔥 News

- [2025/09] [MicroNucML report]( https://www.biorxiv.org/content/10.1101/2025.09.20.677550v1) is released.
- [2025/06] MicroNuclei Detection v1.0 released.
  
## 📦 Installation
This tool need to be installed before use. All the requirements are in `requirements.txt` (for compute canada, use `requirements_ComputeCan.txt`). Please install pytorch and torchvision dependencies. 
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

You can install this tool on a GPU machine using:

```
git clone https://github.com/kew6688/MicroNuclei_Detection.git && cd MicroNuclei_Detection/src
pip install -r requirements.txt
pip install -e .
```

In order to fully utilize the pipeline, we integrated with SAM2 for nuclei segmentation. 

## 📚 Model Checkpoints

First, we need to download a model checkpoint. All the model checkpoints can be downloaded by running:

```
cd MicroNuclei_Detection/src/checkpoints && ./download_ckpts.sh
```

The pre-trained models can be download from huggingface:
- [resnet50](https://huggingface.co/ccglab22/MaskRCNN-resnet50FPN)
- [swin](https://huggingface.co/ccglab22/MaskRCNN-swinFPN)
- [resnet101](https://huggingface.co/ccglab22/MaskRCNN-resnet101)
  
After downloading the model, the usage of the end-to-end pipeline is described below.

download SAM2 checkpoint by running:
```
cd MicroNuclei_Detection/external/sam2/checkpoints && \
./download_ckpts.sh && \
cd ..
```

## 🧩 Usage:
### **Use as a library.** 
This includes:
 - predict counts of micronuclei
 - predict masks
 - output a info dictionary with box, center location and size

Please refer to the examples in the [tutorial.ipynb](./notebooks/tutorial.ipynb) (open in [colab](https://colab.research.google.com/github/kew6688/MicroNuclei_Detection/blob/main/notebooks/tutorial.ipynb))

### **Run the main python script.** 
Automated pipeline to process images. 

```
Example Usage:
      >>> python MicroNuclei_Detection/src/compute_scripts/image_process.py --src ./sample_images --dst test.json 
```
**Input**: The model expects 8-bit RGB images (tif, png, jpg) without any text labels. The training data is 20x magnificent.

**Output**: can be found in the [output](#output-format) section

### Parameters and Arguments
| Parameter          | Short Form | Required | Default    | Type         | Description                                                                                       |
|--------------------|------------|----------|------------|--------------|---------------------------------------------------------------------------------------------------|
| `--src`            | `-s`       | Yes      | N/A        | String       | Pathway to image.                                                                                 |
| `--dst`            | `-d`       | Yes      | N/A        | String       | Pathway to output.                                                                                |
| `--mode`           | `-mod`     | No      | "ALL"        | String       | mode for output. ALL for both nuc and mn, NUC for only nuclei, MN for only micronuclei. Options: `["MN", "NUC", "ALL"]`                                                  |
| `--conf`           | `-c`       | No       | 0.7        | Float        | confidence threshold for micronuclei detection, e.g. --conf 0.4                                   |
| `--parent`         | `-p`       | No       | "edge"        | String        | Parent assign method, use closest center or edge to find nearest parent nuclei, Options: `["center", "edge"]`   |
| `--apop`         | `-apop`       | No       | True        | Bool        | Turn ON/OFF the apoptosis check function   |
| `--apop_cnt`         | `-apop_cnt`       | No       | 5        | Integer        | The threshold to consider MNs cluster to be the apoptosis  |
| `--mask`         | `-mask`       | No       | N/A       | String       | The folder location of input mask for nuclei segmentation, the masks in the folder should have the same name matched to the images. The shape expected is [n, w, h], n is number of nuclei, w,h is image shape |



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
The following is the overview of the dataset. The dataset can be download [here](https://zenodo.org/records/15312291), by downloading the dataset you agree that you have read and accept licence.

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


