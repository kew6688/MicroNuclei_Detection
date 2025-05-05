import os

import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import peak_local_max
import tensorflow as tf

import deepcell
from deepcell.applications import NuclearSegmentation
from deepcell.image_generators import CroppingDataGenerator
from deepcell.losses import weighted_categorical_crossentropy
from deepcell.model_zoo.panopticnet import PanopticNet
from deepcell.utils.train_utils import count_gpus, rate_scheduler
from deepcell_toolbox.deep_watershed import deep_watershed
from deepcell_toolbox.metrics import Metrics
from deepcell_toolbox.processing import histogram_normalization

from PIL import Image

def load_data(filepath, set_nums, test_size=0.2, seed=0):
    X = []
    y = []
    for file in os.listdir(os.path.join(filepath,'final_masks')):
      if file[:2] == '._':
        continue
      im = Image.open(os.path.join(filepath,'images',file[:-4]+".png"))

      # Convert to grayscale
      im = im.convert("L")

      X.append(np.array(im))

      masks = np.load(os.path.join(filepath,'final_masks',file))
      if masks.ndim == 3:
        masks = np.expand_dims(masks, axis=0)

      # Assume `mask` is your input tensor of shape (n, 1, W, H)
      n, _, W, H = masks.shape

      # Remove the singleton dimension to get shape (n, W, H)
      masks = masks.squeeze(1)

      # Assign unique IDs to each object
      for i in range(n):
          masks[i] *= (i + 1)

      # Merge all objects into a single mask with shape (W, H)
      merged_mask = masks.sum(axis=0)
      y.append(merged_mask)

    X = np.expand_dims(np.asarray(X), axis=(3))
    y = np.expand_dims(np.asarray(y), axis=(3))

    # Assume `patches` is your array with shape (n, W, H, 1)
    n, W, H, _ = X.shape

    # Calculate the required padding to go from (W, H) to (256, 256)
    padding = ((0, 0), (16, 16), (16, 16), (0, 0))  # No padding on the batch and channel dimensions

    # Apply padding
    X = np.pad(X, padding, mode='constant', constant_values=0)
    y = np.pad(y, padding, mode='constant', constant_values=0)

    X_train, X_test, y_train, y_test = X[:-100], X[-100:], y[:-100], y[-100:]

    return X_train, X_test, y_train, y_test



if __name__ == "__main__":
    data_dir = 'mnMask/data'
    model_path = 'NuclearSegmentation'
    metrics_path = 'metrics.yaml'
    train_log = 'train_log.csv'
    
    set_nums = 4
    test_size = 0.2
    seed = 7

    X_train, X_test, y_train, y_test = load_data(data_dir, set_nums, test_size=test_size, seed=seed)
    X_val = X_test
    y_val = y_test
    print('X_train.shape: {}\ty_train.shape: {}'.format(X_train.shape, y_train.shape))
    print('X_test.shape: {}\ty_test.shape: {}'.format(X_test.shape, y_test.shape))

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
    epochs = 20
    batch_size = 16
    lr = 1e-4

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

    inputs, outputs = train_data.next()

    img = inputs[0]
    inner_distance = outputs[0]
    outer_distance = outputs[1]
    fgbg = outputs[2]

    input_shape = (crop_size, crop_size, 1)

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

    weights_path = os.path.join(model_path, "model_weights.h5")
    model.save_weights(weights_path, save_format="h5")
    prediction_model = PanopticNet(
        backbone=backbone,
        input_shape=input_shape,
        norm_method=None,
        num_semantic_heads=2,
        num_semantic_classes=[1, 1],  # inner distance, outer distance
        location=location,  # should always be true
        include_top=True,
        backbone_levels=["C1", "C2", "C3", "C4", "C5"],
        pyramid_levels=pyramid_levels,
    )
    prediction_model.load_weights(weights_path, by_name=True)

    # make predictions on testing data
    from timeit import default_timer

    start = default_timer()
    test_images = prediction_model.predict(X_test)
    watershed_time = default_timer() - start

    print('Watershed segmentation of shape', test_images[0].shape,
        'in', watershed_time, 'seconds.')
    
    import random

    from matplotlib import pyplot as plt

    from deepcell_toolbox.deep_watershed import deep_watershed

    index = random.randint(0, X_test.shape[0])
    print(index)

    fig, axes = plt.subplots(1, 4, figsize=(20, 20))

    masks = deep_watershed(
        test_images,
        min_distance=1,
        detection_threshold=0.1,
        distance_threshold=0.01,
        exclude_border=False,
        small_objects_threshold=0)

    # calculated in the postprocessing above, but useful for visualizing
    inner_distance = test_images[0]
    outer_distance = test_images[1]

    # raw image with centroid
    axes[0].imshow(X_test[index, ..., 0])
    axes[0].set_title('Raw')

    axes[1].imshow(inner_distance[index, ..., 0], cmap='jet')
    axes[1].set_title('Inner Distance')
    axes[2].imshow(outer_distance[index, ..., 0], cmap='jet')
    axes[2].set_title('Outer Distance')
    axes[3].imshow(masks[index, ...], cmap='jet')
    axes[3].set_title('Instance Mask')

    plt.savefig("predict.png")

    import numpy as np
    # from skimage.morphology import watershed, remove_small_objects
    from skimage.segmentation import clear_border
    from deepcell.metrics import Metrics

    y_pred = masks.copy()
    y_true = y_test.copy().astype('int')

    m = Metrics('DeepWatershed - Remove no pixels', seg=False)
    m.calc_object_stats(y_true, y_pred)