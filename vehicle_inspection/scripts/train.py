"""
train.py — DeepLabV3+ for 6-class semantic segmentation on CarDD
"""

import os
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.callbacks import History

from dataset import CarDamageDataset
from default_config import CLASSES, DATASET, MODEL_ARTIFACT, MODEL_CONFIG
from model import CustomMeanIoU, build_model
from utils import plot_learning_curves, save_train_history


# pylint: disable=redefined-outer-name
def load_dataset() -> tuple[tf.data.Dataset, int, tf.data.Dataset, int]:
    train_dataset, train_len = CarDamageDataset(
        img_dir=DATASET["train_img_dir"],
        ann_path=DATASET["json_path_train"],
        image_size=MODEL_CONFIG["image_size"],
        batch_size=MODEL_CONFIG["batch_size"],
        num_classes=MODEL_CONFIG["num_classes"],
        shuffle=True,
    ).build_tf_dataset()

    val_dataset, val_len = CarDamageDataset(
        img_dir=DATASET["val_img_dir"],
        ann_path=DATASET["json_path_val"],
        image_size=MODEL_CONFIG["image_size"],
        batch_size=MODEL_CONFIG["batch_size"],
        num_classes=MODEL_CONFIG["num_classes"],
        shuffle=False,
    ).build_tf_dataset(repeat=True)

    return train_dataset, train_len, val_dataset, val_len


def train_model(
    model: tf.keras.Model,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    train_len: int,
    val_len: int,
    epochs: int,
    model_name: str,
) -> History:
    """Train model and return a History object.

    A history attribute containing the lists of successive losses
    and other metrics.
    """
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_name, save_weights_only=False, save_best_only=True, mode="min"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=MODEL_CONFIG["patience_es"], verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        ),
    ]

    steps_per_epoch = train_len // MODEL_CONFIG["batch_size"]
    validation_steps = val_len // MODEL_CONFIG["batch_size"]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )

    return history


if __name__ == "__main__":
    os.makedirs(MODEL_ARTIFACT["model_h5_path"], exist_ok=True)
    os.makedirs(MODEL_ARTIFACT["model_pb_path"], exist_ok=True)
    os.makedirs(MODEL_ARTIFACT["artifacts"], exist_ok=True)

    train_dataset, train_len, val_dataset, val_len = load_dataset()

    model = build_model(
        learning_rate=MODEL_CONFIG["learning_rate"],
        n_classes=MODEL_CONFIG["num_classes"],
    )

    history = train_model(
        model,
        train_dataset,
        val_dataset,
        train_len,
        val_len,
        MODEL_CONFIG["epochs"],
        model_name=os.path.join(MODEL_ARTIFACT["model_h5_path"], "best_model.h5"),
    )

    save_train_history(history)
    plot_learning_curves(history)
