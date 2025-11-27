"""Training module of document detection model."""

import glob
import os
import random
import sys
import uuid
from enum import Enum, auto
from typing import Dict, Optional

import numpy as np
import tensorflow as tf

from deeplab_architecture import DEEP_LAB, deep_lab_v3_plus
from default_config import MODEL_ARCHITECTURE, MODEL_CONFIG, ClassTypes

SEED = 1

_ACTIVATION_FN = "softmax"


class ModelInitialization(Enum):
    """
    Select model initialization weights for training.

    We can choose to initialize with pre-trained weights from
    imagenet or with the current document detection model we have
    in production.
    """

    imagenet = auto()


class TrainableBackbone:
    """
    Class to handle freezing or ungreezing backbone layers.
    """

    def __init__(self, model: tf.keras.Model, freeze: bool):
        self.model = model
        self.freeze = freeze

        if self.freeze:
            self._freeze_backbone()
        else:
            self._unfreeze_backbone()

    def _freeze_backbone(self) -> None:
        """Freeze backbone layers.

        This function freezes the backbone (encoder) layers, thus
        there is no updating on weights for these layers (excluding
        from training).
        """
        for layer in self.model.layers:
            # The model is sequential, meaning that we first
            # pass through input layer and encoder and then we
            # go through the decoder. When we meet the first layer
            # of decoder we stop.
            if DEEP_LAB in layer.name:
                break

            # We do not want to freeze input layer and
            # batch normalization layers
            if isinstance(
                layer, (tf.keras.layers.InputLayer, tf.keras.layers.BatchNormalization)
            ):
                continue

            # Freeze only backbone layers
            layer.trainable = False

    def _unfreeze_backbone(self) -> None:
        """Unfreeze backbone layers."""
        # The decoder is already trainable, so no harm
        # if we set it to True again. We enable encoder
        # to be trainable too.
        for layer in self.model.layers:
            layer.trainable = True


class CustomMeanIoU(tf.keras.metrics.MeanIoU):
    """
    Custom Mean IoU for semantic segmentation.
    Compatible with model serialization/deserialization.
    """

    def __init__(
        self,
        num_classes,
        name="mIoU",
        dtype=tf.float32,
        ignore_class=None,
        sparse_y_true=True,
        sparse_y_pred=True,
        axis=-1,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            name=name,
            dtype=dtype,
            ignore_class=ignore_class,
            sparse_y_true=sparse_y_true,
            sparse_y_pred=sparse_y_pred,
            axis=axis,
            **kwargs,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        return super().update_state(y_true, y_pred, sample_weight)


def set_seeds():
    """
    Set the seed to training stability and reproducibility.
    """
    tf.keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()


def build_model(learning_rate: str, n_classes: int) -> tf.keras.Model:
    """
    Build and load the model.
    """
    set_seeds()

    model = _load_imagenet_pretrained_model(n_classes)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="seg_accuracy"),
        CustomMeanIoU(num_classes=n_classes, name="mIoU"),
    ]

    model.compile(
        optimizer,
        tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False,
        ),
        metrics=metrics,
    )
    return model


def _load_imagenet_pretrained_model(n_classes: int) -> tf.keras.Model:
    """
    Build and load pretrained model with imagenet weights initialization.
    """
    # Create model
    model = deep_lab_v3_plus(
        image_size=MODEL_CONFIG["image_size"], num_classes=n_classes
    )

    return model
