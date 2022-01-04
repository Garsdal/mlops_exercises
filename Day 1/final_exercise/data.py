import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
import os
import matplotlib.pyplot as plt

import helper

path = "../data/corruptmnist"

test_files = ["\\".join([path,os.listdir(path)[0]])]
train_files = ["\\".join([path,x]) for x in os.listdir(path)[1:6]]

data_all = {'images': np.empty([0, 28, 28]), 'labels': np.empty([0,])}

def load_MNIST_files(filenames):
    # We load all .npz in the filenames
    for file in filenames:   
        f = np.load(file)
        
        # We only grab 'images' and 'labels'
        for key in f.files[0:2]:
            data_all[key] = np.concatenate([data_all[key], f[key]], axis = 0)
        
        data = []
        for key in data_all:
            if key == "images":
                data.append(data_all[key].astype(np.float32))
            elif key == "labels":
                data.append(data_all[key].astype(np.int64))
    return tuple(data)

class CorruptedMNISTDataset(Dataset):
    def __init__(self, data_files):
        self.data = load_MNIST_files(data_files)

    def __getitem__(self, index):
        return self.data[0][index], self.data[1][index]

    def __len__(self):
        return len(self.data[0])

def mnist(batch_size = 64):
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

    # exchange with the corrupted mnist dataset
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784) 
    return train_loader, test_loader
