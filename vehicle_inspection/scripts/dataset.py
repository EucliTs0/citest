import random
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from tqdm import tqdm

random.seed(42)


class CarDamageDataset:
    def __init__(
        self,
        img_dir,
        ann_path,
        image_size=256,
        batch_size=8,
        num_classes=6,
        shuffle=True,
    ):
        self.img_dir = Path(img_dir)
        self.coco = COCO(ann_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle

        self.image_ids = list(self.coco.imgs.keys())

    def _load_image_and_mask(self, image_id):
        img_info = self.coco.loadImgs(image_id)[0]
        img_path = str(self.img_dir / img_info["file_name"])

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)

        if not hasattr(self, "cat_to_index"):
            cat_ids = sorted(self.coco.getCatIds())
            self.cat_to_index = {cat_id: i for i, cat_id in enumerate(cat_ids)}

        for ann in anns:
            rle = self.coco.annToMask(ann)
            if rle.shape != mask.shape:
                rle = cv2.resize(rle, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            cat_id = ann["category_id"]
            mapped_id = self.cat_to_index.get(cat_id, 0)
            mask[rle == 1] = mapped_id

        img = cv2.resize(
            img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
        )
        mask = cv2.resize(
            mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST
        )
        mask = np.expand_dims(mask, axis=-1).astype(np.uint8)

        img = img.astype(np.float32) / 255.0
        return img, mask

    def build_tf_dataset(self, repeat: bool = True):
        output_signature = (
            tf.TensorSpec(
                shape=(self.image_size, self.image_size, 3), dtype=tf.float32
            ),
            tf.TensorSpec(shape=(self.image_size, self.image_size, 1), dtype=tf.uint8),
        )

        ds = tf.data.Dataset.from_tensor_slices(self.image_ids)
        if self.shuffle:
            ds = ds.shuffle(
                buffer_size=len(self.image_ids), reshuffle_each_iteration=True
            )

        def _load_and_preprocess(image_id):
            img, mask = tf.py_function(
                func=lambda i: self._load_image_and_mask(int(i.numpy())),
                inp=[image_id],
                Tout=[tf.float32, tf.uint8],
            )
            img.set_shape((self.image_size, self.image_size, 3))
            mask.set_shape((self.image_size, self.image_size, 1))
            return img, mask

        ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        if repeat:
            ds = ds.repeat()

        return ds, len(self.image_ids)
