"""
Deeplab Model Architecture Module

This module defines the DeeplabV3Plus model architecture
for semantic segmentation.
The architecture includes dilated spatial pyramid pooling and
utilizes a ResNet50 backbone.

Reference:
    - https://keras.io/examples/vision/deeplabv3_plus/

Functions:
    - convolution_block: Helper function for creating a convolution block
      with Conv2D, BatchNormalization, and ReLU.

    - dilated_spatial_pyramid_pooling: Function implementing the
      dilated spatial pyramid pooling block.

    - deep_lab_v3_plus: The main function representing the DeeplabV3Plus model.
"""

import uuid

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, ops

from default_config import MODEL_ARCHITECTURE

DEEP_LAB: str = "deep_lab"


def convolution_block(
    block_input: tf.Tensor,
    num_filters: int = MODEL_ARCHITECTURE["num_filters"],
    kernel_size: int = MODEL_ARCHITECTURE["kernel_size"],
    dilation_rate: int = MODEL_ARCHITECTURE["dilation_rate"],
    padding: str = MODEL_ARCHITECTURE["padding"],
    use_bias: bool = MODEL_ARCHITECTURE["use_bias"],
) -> tf.Tensor:
    """
    Convolutional block with Conv2D, BatchNormalization, and ReLU activation.

    Args:
        block_input (tf.Tensor): Input tensor.
        num_filters (int).
        kernel_size (int).
        dilation_rate(int).
        padding (str).
        use_bias (bool).

    Returns:
        tf.Tensor: Output tensor.
    """
    conv_layer = layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
        name=DEEP_LAB + str(uuid.uuid4()),
    )(block_input)
    normalized_layer = layers.BatchNormalization()(conv_layer)
    return layers.ReLU()(normalized_layer)


def dilated_spatial_pyramid_pooling(dspp_input: tf.Tensor) -> tf.Tensor:
    """
    Dilated Spatial Pyramid Pooling block.

    Args:
        dspp_input (tf.Tensor): Input tensor.

    Returns:
        tf.Tensor: Output tensor.
    """
    dims = dspp_input.shape
    # Average pooling to reduce spatial dimensions
    avg_pooled = layers.AveragePooling2D(
        pool_size=(dims[-3], dims[-2]), name=DEEP_LAB + str(uuid.uuid4())
    )(dspp_input)
    avg_pooled_conv_block = convolution_block(avg_pooled, kernel_size=1, use_bias=True)
    # Upsample to original spatial dimensions
    upsampled_pool = layers.UpSampling2D(
        size=(
            dims[-3] // avg_pooled_conv_block.shape[1],
            dims[-2] // avg_pooled_conv_block.shape[2],
        ),
        interpolation=MODEL_ARCHITECTURE["interpolation"],
    )(avg_pooled_conv_block)

    # Dilated convolution blocks with different rates
    dilated_blocks = [
        convolution_block(dspp_input, kernel_size=kernel, dilation_rate=dilation)
        for kernel, dilation in zip(
            MODEL_ARCHITECTURE["dilated_block_kernel_sizes"],
            MODEL_ARCHITECTURE["dilated_block_dilation_rates"],
        )
    ]

    # Concatenate all blocks
    concatenated_blocks = layers.Concatenate(axis=-1)([upsampled_pool] + dilated_blocks)
    # Final convolution block
    output = convolution_block(concatenated_blocks, kernel_size=1)
    return output


def deep_lab_v3_plus(image_size: int, num_classes: int) -> keras.Model:
    """
    DeeplabV3Plus model architecture.

    Args:
        image_size (int): Input image size.
        num_classes (int): Number of output classes.

    Returns:
        keras.Model: DeeplabV3Plus model.
    """
    model_input = keras.Input(shape=(image_size, image_size, 3))
    resnet50 = keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )

    resnet_block_1_output = resnet50.get_layer(
        MODEL_ARCHITECTURE["resnet_block_name_1"]
    ).output
    dspp_output = dilated_spatial_pyramid_pooling(resnet_block_1_output)

    # Upsample and process another input block
    upsampled_dspp = layers.UpSampling2D(
        size=(
            image_size
            // MODEL_ARCHITECTURE["image_size_reduction_factor"]
            // dspp_output.shape[1],
            image_size
            // MODEL_ARCHITECTURE["image_size_reduction_factor"]
            // dspp_output.shape[2],
        ),
        interpolation=MODEL_ARCHITECTURE["interpolation"],
        name=DEEP_LAB + str(uuid.uuid4()),
    )(dspp_output)

    resnet_block_2_output = resnet50.get_layer(
        MODEL_ARCHITECTURE["resnet_block_name_2"]
    ).output
    resnet_block_2_processed = convolution_block(
        resnet_block_2_output,
        num_filters=MODEL_ARCHITECTURE["num_resnet_filters"],
        kernel_size=1,
    )

    # Concatenate processed inputs
    concatenated_inputs = layers.Concatenate(
        axis=-1, name=DEEP_LAB + str(uuid.uuid4())
    )([upsampled_dspp, resnet_block_2_processed])
    concatenated_inputs_conv = convolution_block(concatenated_inputs)
    concatenated_inputs_conv_2 = convolution_block(concatenated_inputs_conv)

    upsampled_final = layers.UpSampling2D(
        size=(
            image_size // concatenated_inputs_conv_2.shape[1],
            image_size // concatenated_inputs_conv_2.shape[2],
        ),
        interpolation=MODEL_ARCHITECTURE["interpolation"],
        name=DEEP_LAB + str(uuid.uuid4()),
    )(concatenated_inputs_conv_2)

    # Final convolution to get the output classes
    model_output = layers.Conv2D(
        num_classes,
        kernel_size=MODEL_ARCHITECTURE["final_kernel_size"],
        padding=MODEL_ARCHITECTURE["final_padding"],
        name=DEEP_LAB + str(uuid.uuid4()),
    )(upsampled_final)
    model_output = layers.Activation(
        MODEL_ARCHITECTURE["final_activation"], name=DEEP_LAB + str(uuid.uuid4())
    )(model_output)
    return keras.Model(inputs=model_input, outputs=model_output)
