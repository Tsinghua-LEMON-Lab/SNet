# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# @Filename: __init__.py
# @Date: 2019-04-18-12-21
# @Author: Nuullll (Yilong Guo)
# @Email: vfirst218@gmail.com


import os
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data.dataset import Subset


DATASET_DIR = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)


class MNISTLoader(object):
    """
    Dataset loader for MNIST.
    """

    def __init__(self, options):

        self.options = options

        # re-scale
        self._rescale()

        # filter categories
        self._filter_categories()

    def _rescale(self):
        size = self.options.get('size', (28, 28))

        self.training_set = MNIST(DATASET_DIR, train=True, download=True, transform=transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ]))

        self.testing_set = MNIST(DATASET_DIR, train=False, download=True, transform=transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ]))

    def _filter_categories(self):
        include_categories = self.options.get('include_categories', list(range(10)))

        def _filter(dataset):

            include_indices = []

            for i, (image, label) in enumerate(dataset):
                if label in include_categories:
                    include_indices.append(i)

            return Subset(dataset, include_indices)

        self.training_set = _filter(self.training_set)
        self.testing_set = _filter(self.testing_set)
