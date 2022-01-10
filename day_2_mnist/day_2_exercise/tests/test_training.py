import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset

from src.data.data import mnist
from src.models.model import FFNN, CNN
import pytest
import numpy as np

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
# We assert that we have obtained a training loss (maybe not the best test but would indicate that our loop runs)
def test_training():
    # We manually specify the model parameters for now
    num_classes = 10
    channels = 1
    height = 28
    width = 28
    num_filters = 16
    model = CNN(num_classes, channels, height, width, num_filters)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # We setup the dataloaders
    batch_size = 1
    train_files = ['data/processed/train_images.pt', 'data/processed/train_labels.pt']
    test_files = ['data/processed/test_images.pt', 'data/processed/test_labels.pt']
    train_loader, test_loader, train_set, test_set = mnist(train_files, test_files)
    images, labels = next(iter(train_loader))

    # We run a single iteration of our training loop
    steps = 0
    running_loss = 0
    print_every = 1

    train_loss = []
    for e in range(1):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in train_loader:
            steps += 1

            # Format the image into [batch, channel, dim, dim]
            images = images.view(images.shape[0], 1, images.shape[1], images.shape[2])

            optimizer.zero_grad()
            
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loss.append(loss)

            if steps % print_every == 0:
                # Make sure dropout and grads are on for training
                model.train()

    assert loss != 0, "The training loop does not output a training loss."