"""
Evaluate the performance of deep cell's pretrained segmentation model
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
import tensorflow as tf

from deepcell.applications import NuclearSegmentation
from deepcell.image_generators import CroppingDataGenerator
from deepcell.losses import weighted_categorical_crossentropy
from deepcell.model_zoo.panopticnet import PanopticNet
from deepcell.utils.train_utils import count_gpus, rate_scheduler
from deepcell_toolbox.deep_watershed import deep_watershed
from deepcell_toolbox.metrics import Metrics
from deepcell_toolbox.processing import histogram_normalization

from deepcell.applications import Mesmer
app = Mesmer()

segmentation_predictions_nuc = app.predict(X_train, image_mpp=0.5, compartment='nuclear')

overlay_data_nuc = make_outline_overlay(rgb_data=rgb_images, predictions=segmentation_predictions_nuc)