import argparse
import sys

import torch
from torch import nn, optim

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from src.data.data import mnist
from src.models.model import CNN

class Visualize(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def visualize(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        #parser.add_argument('load_model_from', default="")
        parser.add_argument('--ckpt', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])

        # We load the specified model based on the given args.ckpt
        ckpt_path = args.ckpt
        model = self.load_checkpoint_CNN(ckpt_path)
        print(f"Model was loaded from {ckpt_path}.")

        # We visualize
        X = model.state_dict()['l_1.weight']
        X_embedded = TSNE(n_components=2, init='random').fit_transform(X)

        # We output a figure
        plt.figure()
        plt.scatter(X_embedded[:,0], X_embedded[:,1], color = 'g', label='Feature l_1 after TSNE embedding', zorder=1)
        plt.legend()

        # We don't want to input more arguments so we extract the model folder in the reports from the ckpt
        model_folder = args.ckpt.split("/")[1]
        print(model_folder)
        fig_path = "//".join(['reports/figures', model_folder, 'png', 'l_1_weights.png'])

        # We save the training curve
        print("L_1 model weights after TSNE embedding was saved at:", fig_path)
        plt.savefig(fig_path, dpi=500)
     
    def load_checkpoint_CNN(self, filepath):
        print(filepath)
        checkpoint = torch.load(filepath)
        model = CNN(checkpoint['num_classes'],
                                checkpoint['channels'],
                                checkpoint['height'],
                                checkpoint['width'],
                                checkpoint['num_filters'])
        model.load_state_dict(checkpoint['state_dict'])
        
        return model

if __name__ == '__main__':
    Visualize()
