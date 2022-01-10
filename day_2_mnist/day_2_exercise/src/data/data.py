import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import os
import matplotlib.pyplot as plt

def load_images_labels(data_files):
            data = []
            for file in data_files:
                data.append(torch.load(file))
            return(tuple(data))

class CorruptedMNISTDataset(Dataset):
    def __init__(self, data_files):
        self.data = load_images_labels(data_files)

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return len(self.data[0])

def mnist(train_files, test_files, batch_size = 64):
        train_set = CorruptedMNISTDataset(train_files)
        test_set = CorruptedMNISTDataset(test_files)

        train_loader = DataLoader(dataset=train_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

        test_loader = DataLoader(dataset=test_set,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

        return(train_loader, test_loader, train_set, test_set)