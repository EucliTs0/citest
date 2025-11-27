"""Configuration module."""

ClassTypes = dict[str, str | list[float]]

ModelArchTypes = int | str | bool | tuple | list


MODEL_CONFIG: dict[str, int | float] = {
    "image_size": 256,
    "num_classes": 6,
    "batch_size": 16,
    "learning_rate": 1e-4,
    "epochs": 50,
    "patience_es": 10,
}


MODEL_ARCHITECTURE: dict[str, ModelArchTypes] = {
    "num_filters": 128,
    "kernel_size": 3,
    "dilation_rate": 1,
    "padding": "same",
    "use_bias": False,
    "interpolation": "bilinear",
    "image_size_reduction_factor": 4,
    "resnet_block_name_1": "conv4_block6_2_relu",
    "resnet_block_name_2": "conv2_block3_2_relu",
    "num_resnet_filters": 24,
    "final_kernel_size": (1, 1),
    "final_padding": "same",
    "final_activation": "softmax",
    "dilated_block_kernel_sizes": [1, 3, 3, 3],
    "dilated_block_dilation_rates": [1, 6, 12, 18],
}


MODEL_ARTIFACT: dict[str, str] = {
    "model_h5_path": "model/h5/",
    "model_pb_path": "model/pb/",
    "artifacts": "artifacts",
}


DATASET: dict[str, str] = {
    "train_img_dir": "/home/dim/workspace/ml-lab-vehicle-inspection/CarDD_release/CarDD_COCO/train2017",
    "val_img_dir": "/home/dim/workspace/ml-lab-vehicle-inspection/CarDD_release/CarDD_COCO/val2017",
    "test_img_dir": "/home/dim/workspace/ml-lab-vehicle-inspection/CarDD_release/CarDD_COCO/test2017",
    "json_path_train": "/home/dim/workspace/ml-lab-vehicle-inspection/CarDD_release/CarDD_COCO/annotations/instances_train2017.json",
    "json_path_val": "/home/dim/workspace/ml-lab-vehicle-inspection/CarDD_release/CarDD_COCO/annotations/instances_val2017.json",
    "json_path_test": "/home/dim/workspace/ml-lab-vehicle-inspection/CarDD_release/CarDD_COCO/annotations/instances_test2017.json",
}


CLASSES: dict[str, ClassTypes] = {
    "0": {"name": "DENT", "color": [128, 128, 0], "color_name": "OLIVE_GREEN"},
    "1": {"name": "SCRATCH", "color": [255, 0, 0], "color_name": "RED"},
    "2": {"name": "CRACK", "color": [0, 255, 0], "color_name": "GREEN"},
    "3": {"name": "GLASS_SHATTER", "color": [0, 0, 255], "color_name": "BLUE"},
    "4": {"name": "LAMP_BROKEN", "color": [255, 255, 0], "color_name": "YELLOW"},
    "5": {"name": "TIRE_FLAT", "color": [255, 0, 255], "color_name": "MAGENTA"},
}
