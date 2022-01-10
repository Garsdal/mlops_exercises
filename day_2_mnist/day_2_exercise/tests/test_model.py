import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pytest

from src.data.data import mnist
from src.models.model import FFNN, CNN
import numpy as np

train_files = ['data/processed/train_images.pt', 'data/processed/train_labels.pt']
test_files = ['data/processed/test_images.pt', 'data/processed/test_labels.pt']

train_files = ['data/interim/train_images.pt', 'data/interim/train_labels.pt']
test_files = ['data/interim/test_images.pt', 'data/interim/test_labels.pt']

check = False
# We check for files
bools = []
for file in (train_files + test_files):
    bools.append(os.path.isfile(file))

check = all(np.array(bools))
# We wrap the asserts in a function starting with test_ for the pytest
@pytest.mark.skipif(~check, reason="Data files not found")
# No matter the number of filters in the CNN we should always return num_classes
@pytest.mark.parametrize("test_input,expected", [(16, 10), (8, 10), (4, 10)])
def test_model(test_input, expected):
    # We setup the dataloaders
    batch_size = 64
    train_loader, test_loader, train_set, test_set = mnist(train_files, test_files)
    images, labels = next(iter(train_loader))

    # We manually specify the model parameters for now
    num_classes = expected
    channels = 1
    height = 28
    width = 28
    num_filters = test_input
    model = CNN(num_classes, channels, height, width, num_filters)

    # We forward pass an image and assert the input/output shape
    images = images.view(images.shape[0], 1, images.shape[1], images.shape[2])
    output = model.forward(images)

    # We assert the output shape of the model after forward passing an images
    assert output[0,:].shape == torch.Size([num_classes]), "The output dimensions does not correspond to num_classes."

# We didn't increase coverage since this refers to FFNN model class which is legacy
# pytest tests/