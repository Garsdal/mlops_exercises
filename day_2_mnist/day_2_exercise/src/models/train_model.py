import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader, Dataset

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime as dt
import os
import hydra
from omegaconf import OmegaConf

# Own imports
from src.models.model import FFNN, CNN
from src.data.data import mnist

# Fix SKlearn
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def setup_folders(folder, subfolder):
    if not os.path.exists(folder + subfolder):
        #logging.info(f"{folder + 'png'} does not exist... creating")
        os.makedirs(folder + subfolder)

# The stuff commented out below is for old Hydra code #

# We create a function to init training from the yaml file
#@hydra.main(config_name='train.yaml')
def train():
    #print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
    #train_params = cfg

    # We define training params to be used in the script
    #epochs = train_params['n_epochs']
    #batch_size = train_params['batch_size']
    #lr = train_params['lr']
    #seed = train_params['seed']

    #print("wd", os.getcwd())

    # We still have some leftover arguments
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--lr', default=0.001)
    parser.add_argument('--epochs', default=5)
    parser.add_argument('--path_out_model', default='models/')
    parser.add_argument('--path_out_fig', default='reports/figures/')
    # # add any additional argument that you want
    args = parser.parse_args(sys.argv[2:])
    print(args)

    MODEL_TYPE = 'corrupted_mnist'
    RUN_TIME = dt.now().strftime("%m%d_%H%M_%S")
    save_folder_model = f'{args.path_out_model}/{MODEL_TYPE}_model_{RUN_TIME}/'
    setup_folders(save_folder_model, 'ckpt')

    # We set a seed
    torch.manual_seed(123)

    # HERE WE CREATE TRAINLOADER AND TESTLOADER FROM PROCESSED DATA | we made a data.py file to save these
    batch_size = 64
    train_files = ['data/processed/train_images.pt', 'data/processed/train_labels.pt']
    test_files = ['data/processed/test_images.pt', 'data/processed/test_labels.pt']
    trainloader, testloader, train_set, test_set = mnist(train_files, test_files, batch_size = batch_size)
    
    # We actually don't need all of the below since we give this in model.yaml | not the best implementation
    # For pytorch lightning we should make the CNN class more customizable and put these hyperparams in model.yaml
    images, labels = next(iter(trainloader))

    # Reshape an image to the correct size
    images = images.view(batch_size, 1, images.shape[1], images.shape[2])

    # hyperparameters of the model
    num_classes = 10
    channels = images.shape[1]
    height = images.shape[2]
    width = images.shape[3]
    num_filters = 16

    # Define model
    model = CNN(num_classes, channels, height, width, num_filters)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    running_loss = 0
    print_every = 64

    train_loss = []
    for e in range(int(args.epochs)):
        # Model in training mode, dropout is on
        model.train()
        for images, labels in trainloader:
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
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients for validation, will speed up inference
                with torch.no_grad():
                    test_loss, accuracy = validation(model, testloader, criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, args.epochs),
                    "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                    "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                    "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                
                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()

    # Finally we produce a single plot with the training curve
    save_folder_fig = f'{args.path_out_fig}/{MODEL_TYPE}_model_{RUN_TIME}/'
    setup_folders(save_folder_fig, 'png')

    with torch.no_grad():
        plt.figure()
        plt.plot(train_loss, 'g', label='Training loss', zorder=1)
        plt.legend()
        fig_path = "//".join([save_folder_fig, 'png', 'training_curve.png'])
        plt.savefig(fig_path, dpi=500)

    # We save the training curve
    print("Training curve was saved at:", fig_path)

    # We extract information from the model
    print(model.parameters())

    # CNN checkpoint
    checkpoint = {'num_classes': num_classes,
                    'channels': channels,
                    'height': height,
                    'width': width,
                    'num_filters' : num_filters,
                    'state_dict': model.state_dict()}

    # Save checkpoint
    save_folder_ckpt = f'{args.path_out_model}/{MODEL_TYPE}_model_{RUN_TIME}/ckpt'
    #setup_folders(save_folder_ckpt) # we have already set this up in the model run
    ckpt_path = "//".join([save_folder_ckpt, 'checkpoint_CNN.pth'])
    torch.save(checkpoint, ckpt_path)

    # We save the training curve
    print("Model checkpoint was saved at:", ckpt_path)

def validation(model, testloader, criterion, eval = False):
    accuracy = 0
    test_loss = 0
    for images, labels in testloader:
        # We reshape into [batch, channel, dim, dim]
        images = images.view(images.size()[0], 1, images.shape[1], images.shape[2])

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ## Calculating the accuracy 
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()

        # I changed this here to = for the code to run 
        if eval:
            accuracy = equality.type_as(torch.FloatTensor()).mean()

    return test_loss, accuracy 

# We create a function to create the model from the yaml file
# @hydra.main(config_name='model.yaml')
# def create_model(cfg):
#     print(f"configuration: \n {OmegaConf.to_yaml(cfg)}")
#     model_params = cfg
    
#     # We define model params to be used in the script | We could have put more in depth stuff in the model.yaml file but we will do that later
#     # We are actually breaking the dynamic part of the code by hard coding the values in a config file but it works for now
#     num_classes = model_params['num_classes']
#     channels = model_params['channels']
#     height = model_params['height']
#     width = model_params['width']

#     model = CNN(num_classes, channels, height, width)

#     return model, [num_classes, channels, height, width]

if __name__ == '__main__':
    train()