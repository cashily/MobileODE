#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

import argparse

from data.datasets import DATASET_REGISTRY
from data.datasets.classification.base_image_classification_dataset import (
    BaseImageClassificationDataset,
)

@DATASET_REGISTRY.register(name="Dataset_R", type="classification")
class Dataset_R(BaseImageClassificationDataset):
    """ImageNetR dataset, a distribution shift of ImageNet.

    ImageNet-R(endition) contains art, cartoons, deviantart, graffiti, embroidery,
    graphics, origami, paintings, patterns, plastic objects, plush objects, sculptures,
    sketches, tattoos, toys, and video game renditions of ImageNet classes.

    @article{hendrycks2021many,
    title={The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution
    Generalization},
    author={Dan Hendrycks and Steven Basart and Norman Mu and Saurav Kadavath and Frank
    Wang and Evan Dorundo and Rahul Desai and Tyler Zhu and Samyak Parajuli and Mike Guo
    and Dawn Song and Jacob Steinhardt and Justin Gilmer},
    journal={ICCV},
    year={2021}
    }

    """

    def __init__(
        self,
        opts: argparse.Namespace,
        *args,
        **kwargs,
    ) -> None:
        BaseImageClassificationDataset.__init__(
            self,
            opts=opts,
            *args,
            **kwargs,
        )
