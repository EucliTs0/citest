"""
tune_optuna.py — Hyperparameter tuning for DeepLabV3+ on CarDD using Optuna
"""

import os

import optuna
import tensorflow as tf
from dataset_coco import CarDamageDataset
from keras import layers, ops
from tensorflow import keras

tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


class MeanIoUWrapper(keras.metrics.MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.squeeze(y_true, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


# Global cache
_GLOBAL_DATA_CACHE = {}


def get_datasets(batch_size):
    key = f"batch_{batch_size}"
    if key not in _GLOBAL_DATA_CACHE:
        print(f"📦 Loading datasets for batch_size={batch_size}")
        train_dataset, train_len = CarDamageDataset(
            train_img_dir,
            json_path_train,
            image_size=IMAGE_SIZE,
            batch_size=batch_size,
            num_classes=NUM_CLASSES,
            shuffle=True,
        ).build_tf_dataset()
        val_dataset, val_len = CarDamageDataset(
            val_img_dir,
            json_path_val,
            image_size=IMAGE_SIZE,
            batch_size=batch_size,
            num_classes=NUM_CLASSES,
            shuffle=False,
        ).build_tf_dataset()
        _GLOBAL_DATA_CACHE[key] = (train_dataset, train_len, val_dataset, val_len)
    return _GLOBAL_DATA_CACHE[key]


# ============================================================
# Config
# ============================================================

IMAGE_SIZE = 256
NUM_CLASSES = 6
EPOCHS = 10  # keep small for tuning
train_img_dir = (
    "/home/dim/workspace/ml-lab-vehicle-inspection/CarDD_release/CarDD_COCO/train2017"
)
val_img_dir = (
    "/home/dim/workspace/ml-lab-vehicle-inspection/CarDD_release/CarDD_COCO/val2017"
)
json_path_train = "/home/dim/workspace/ml-lab-vehicle-inspection/CarDD_release/CarDD_COCO/annotations/instances_train2017.json"
json_path_val = "/home/dim/workspace/ml-lab-vehicle-inspection/CarDD_release/CarDD_COCO/annotations/instances_val2017.json"


# ============================================================
# Model builder (parametrized)
# ============================================================


def convolution_block(
    x, num_filters=256, kernel_size=3, dilation_rate=1, use_bias=False
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(x)
    x = layers.BatchNormalization()(x)
    return ops.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input, num_filters=256):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, num_filters=num_filters, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear"
    )(x)
    outs = [
        convolution_block(
            dspp_input, num_filters=num_filters, kernel_size=k, dilation_rate=d
        )
        for k, d in [(1, 1), (3, 6), (3, 12), (3, 18)]
    ]
    x = layers.Concatenate(axis=-1)([out_pool] + outs)
    return convolution_block(x, num_filters=num_filters, kernel_size=1)


def DeeplabV3Plus(image_size, num_classes, base_trainable=True, decoder_filters=256):
    inputs = keras.Input(shape=(image_size, image_size, 3))
    base_model = keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=inputs
    )
    base_model.trainable = base_trainable

    x = base_model.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x, num_filters=decoder_filters)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = base_model.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x, num_filters=decoder_filters)
    x = convolution_block(x, num_filters=decoder_filters)

    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)

    seg_output = layers.Conv2D(num_classes, kernel_size=1, activation="softmax")(x)
    return keras.Model(inputs=inputs, outputs=seg_output)


# ============================================================
# Objective function for Optuna
# ============================================================


def objective(trial):
    tf.keras.backend.clear_session()

    lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16])
    decoder_filters = trial.suggest_categorical("decoder_filters", [128, 256])

    train_dataset, train_len, val_dataset, val_len = get_datasets(batch_size)

    # train_dataset, train_len = CarDamageDataset(
    #     train_img_dir, json_path_train, image_size=IMAGE_SIZE,
    #     batch_size=batch_size, num_classes=NUM_CLASSES, shuffle=True
    # ).build_tf_dataset()
    # val_dataset, val_len = CarDamageDataset(
    #     val_img_dir, json_path_val, image_size=IMAGE_SIZE,
    #     batch_size=batch_size, num_classes=NUM_CLASSES, shuffle=False
    # ).build_tf_dataset()

    steps_per_epoch = train_len // batch_size
    validation_steps = val_len // batch_size

    # Build model
    model = DeeplabV3Plus(IMAGE_SIZE, NUM_CLASSES, True, decoder_filters)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="seg_accuracy"),
        ],
    )

    # Train
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
    )
    val_loss = history.history["val_loss"][-1]
    # val_miou = history.history["val_mean_io_u"][-1]
    print(
        f"Trial {trial.number}: val_loss={val_loss:.4f} | lr={lr:.1e}, batch={batch_size}, filters={decoder_filters}, trainable={True}"
    )
    return val_loss


# %%
# ============================================================
# Run Study
# ============================================================


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)  # try 10–30 depending on compute

print("\nBest trial:")
trial = study.best_trial
print(f"  Value (Mean IoU): {trial.value:.4f}")
for key, val in trial.params.items():
    print(f"  {key}: {val}")

study.trials_dataframe().to_csv("optuna_results.csv", index=False)
