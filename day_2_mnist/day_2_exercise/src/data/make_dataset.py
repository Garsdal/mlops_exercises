# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
#from dotenv import find_dotenv, load_dotenv
import os
import numpy as np
import torch
import torch.nn.functional as F

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # We grab all filenames
    test_files = ["\\".join([input_filepath,os.listdir(input_filepath)[0]])]
    train_files = ["\\".join([input_filepath,x]) for x in os.listdir(input_filepath)[1:6]]

    test_images, test_labels = load_MNIST_files(test_files)
    train_images, train_labels = load_MNIST_files(train_files)

    output_names = ['train_images', 'train_labels', 'test_images', 'test_labels']
    datalist_raw = [train_images, train_labels, test_images, test_labels]
    datalist_processed = []
    for cnt, data in enumerate(datalist_raw):
        data_processed = numpy_to_tensor_normalize(data)
        datalist_processed.append(data_processed)

        # we save the file to processed/filename.pt
        torch.save(data_processed, "//".join([output_filepath, f'{output_names[cnt]}.pt']))

def load_MNIST_files(filenames):
    data_all = {'images': np.empty([0, 28, 28]), 'labels': np.empty([0,])}
    # We load all .npz in the filenames
    for file in filenames:   
        f = np.load(file)
        
        # We only grab 'images' and 'labels'
        for key in f.files[0:2]:
            data_all[key] = np.concatenate([data_all[key], f[key]], axis = 0)
        
        # For some reason images must be float32 and labels int64 or torch crashes later on a float and Long error
        merged_images = data_all[f.files[0]].astype(np.float32)
        merged_labels = data_all[f.files[1]].astype(np.int64)

    return merged_images, merged_labels

def numpy_to_tensor_normalize(data):
    # We dont normalize since our images are already between 0 and 1
    data = torch.tensor(data)
    print(data.shape)
    # We also don't normalize our labels
    #print("Max of the data:", torch.max(data)[0])

    return data

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
