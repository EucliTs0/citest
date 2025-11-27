"""Convert H5 model to .pb."""

from typing import Any

import tensorflow as tf

TARGET_SIZE = 256


def image_preprocessing(image_str_tensor: tf.Tensor) -> tf.Tensor:
    """Pre-processing for image tensors.

    This implements the standard preprocessing that needs
    to be applied to the image tensors before passing them to
    the model. This is used for all input types.
    """
    image = tf.image.decode_jpeg(image_str_tensor, channels=3)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize(image, [TARGET_SIZE, TARGET_SIZE])
    image = tf.squeeze(image, axis=[0])
    image = tf.cast(image, dtype=tf.uint8)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image


def serving_input_receiver_fn() -> tf.Tensor:
    """Define input and preprocessing of a model."""
    # A string that represents base64 encoded image
    input_ph = tf.keras.Input(dtype=tf.string, shape=())

    images_tensor = tf.map_fn(
        image_preprocessing, input_ph, back_prop=False, dtype=tf.float32
    )

    # These two names are required for input of estimator
    return tf.estimator.export.ServingInputReceiver(
        {"input_1": images_tensor}, {"image_bytes": input_ph}
    )


def h5_to_pb(
    h5_model_input_file: str,
    pb_model_output_path: str,
    custom_objects: dict[str, Any] | None = None,
) -> None:
    """Convert h5 model to pb."""
    # The following custom objects correspond to the network architecture
    # we use to train. If we change model architecture, then we need to
    # take care of the custom objects.
    model = tf.keras.models.load_model(
        h5_model_input_file, compile=True, custom_objects=custom_objects
    )

    estimator = tf.keras.estimator.model_to_estimator(
        keras_model=model, model_dir=pb_model_output_path
    )
    estimator.export_saved_model(
        pb_model_output_path, serving_input_receiver_fn=serving_input_receiver_fn
    )
