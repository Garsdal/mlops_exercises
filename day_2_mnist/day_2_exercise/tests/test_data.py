import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
import pytest
import os
import numpy as np

from src.data.data import mnist

train_files = ['data/processed/train_images.pt', 'data/processed/train_labels.pt']
test_files = ['data/processed/test_images.pt', 'data/processed/test_labels.pt']

skip = True
# We check for files
bools = []
for file in (train_files + test_files):
    bools.append(os.path.isfile(file))

skip = not all(np.array(bools))

# We wrap the asserts in a function starting with test_ for the pytest
@pytest.mark.skipif(skip, reason="Data files not found")
def test_data():

    batch_size = 1000
    train_loader, test_loader, train_set, test_set = mnist(train_files, test_files)

    images, labels = next(iter(train_loader))

    # We make assertions for MNIST data
    assert len(train_set) == 25000, "Train set length is not 25000."
    assert len(test_set) == 5000, "Test set length is not 5000."

    # We make assertions for the data shape
    assert images[0,:,:].shape == torch.Size([28, 28]), "The image is not 28x28."

    # We make assertions for the unique labels
    assert torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) in torch.unique(labels), "Not all labels are represented."

# python -m tests.test_data
# Set PYTHONPATH=%cd%