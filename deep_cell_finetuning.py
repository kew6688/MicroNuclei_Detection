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

data_dir = '/notebooks/data'
model_path = 'NuclearSegmentation'
metrics_path = 'metrics.yaml'
train_log = 'train_log.csv'

with np.load(os.path.join(data_dir, 'train.npz')) as data:
    X_train = data['X']
    y_train = data['y']

with np.load(os.path.join(data_dir, 'val.npz')) as data:
    X_val = data['X']
    y_val = data['y']

with np.load(os.path.join(data_dir, 'test.npz')) as data:
    X_test = data['X']
    y_test = data['y']

# Model architecture
backbone = "efficientnetv2bl"
location = True
pyramid_levels = ["P1","P2","P3","P4","P5","P6","P7"]
# Augmentation and transform parameters
seed = 0
min_objects = 1
zoom_min = 0.75
crop_size = 256
outer_erosion_width = 1
inner_distance_alpha = "auto"
inner_distance_beta = 1
inner_erosion_width = 0
# Post processing parameters
maxima_threshold = 0.1
interior_threshold = 0.01
exclude_border = False
small_objects_threshold = 0
min_distance = 10
# Training configuration
epochs = 16
batch_size = 16
lr = 1e-4
input_shape = (crop_size, crop_size, 1)

# data augmentation parameters
zoom_max = 1 / zoom_min

# Preprocess the data
X_train = histogram_normalization(X_train)
X_val = histogram_normalization(X_val)

# use augmentation for training but not validation
datagen = CroppingDataGenerator(
    rotation_range=180,
    zoom_range=(zoom_min, zoom_max),
    horizontal_flip=True,
    vertical_flip=True,
    crop_size=(crop_size, crop_size),
)

datagen_val = CroppingDataGenerator(
    crop_size=(crop_size, crop_size)
)
transforms = ["inner-distance", "outer-distance", "fgbg"]

transforms_kwargs = {
    "outer-distance": {"erosion_width": outer_erosion_width},
    "inner-distance": {
        "alpha": inner_distance_alpha,
        "beta": inner_distance_beta,
        "erosion_width": inner_erosion_width,
    },
}

train_data = datagen.flow(
    {'X': X_train, 'y': y_train},
    seed=seed,
    min_objects=min_objects,
    transforms=transforms,
    transforms_kwargs=transforms_kwargs,
    batch_size=batch_size,
)

print("Created training data generator.")

val_data = datagen_val.flow(
    {'X': X_val, 'y': y_val},
    seed=seed,
    min_objects=min_objects,
    transforms=transforms,
    transforms_kwargs=transforms_kwargs,
    batch_size=batch_size,
)

print("Created validation data generator.")

model = PanopticNet(
    backbone=backbone,
    input_shape=input_shape,
    norm_method=None,
    num_semantic_classes=[1, 1, 2],  # inner distance, outer distance, fgbg
    location=location,
    include_top=True,
    backbone_levels=["C1", "C2", "C3", "C4", "C5"],
    pyramid_levels=pyramid_levels,
)

def semantic_loss(n_classes):
    def _semantic_loss(y_pred, y_true):
        if n_classes > 1:
            return 0.01 * weighted_categorical_crossentropy(
                y_pred, y_true, n_classes=n_classes
            )
        return tf.keras.losses.MSE(y_pred, y_true)

    return _semantic_loss

loss = {}

# Give losses for all of the semantic heads
for layer in model.layers:
    if layer.name.startswith("semantic_"):
        n_classes = layer.output_shape[-1]
        loss[layer.name] = semantic_loss(n_classes)

optimizer = tf.keras.optimizers.Adam(lr=lr, clipnorm=0.001)

model.compile(loss=loss, optimizer=optimizer)

# Clear clutter from previous TensorFlow graphs.
tf.keras.backend.clear_session()

monitor = "val_loss"

csv_logger = tf.keras.callbacks.CSVLogger(train_log)

# Create callbacks for early stopping and pruning.
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor=monitor,
        save_best_only=True,
        verbose=1,
        save_weights_only=False,
    ),
    tf.keras.callbacks.LearningRateScheduler(rate_scheduler(lr=lr, decay=0.99)),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.1,
        patience=5,
        verbose=1,
        mode="auto",
        min_delta=0.0001,
        cooldown=0,
        min_lr=0,
    ),
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    csv_logger,
]

print(f"Training on {count_gpus()} GPUs.")

# Train model.
history = model.fit(
    train_data,
    steps_per_epoch=train_data.y.shape[0] // batch_size,
    epochs=epochs,
    validation_data=val_data,
    validation_steps=val_data.y.shape[0] // batch_size,
    callbacks=callbacks,
)

print("Final", monitor, ":", history.history[monitor][-1])