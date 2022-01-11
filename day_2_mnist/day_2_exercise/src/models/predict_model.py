import argparse
import sys

import torch
from torch import nn, optim

from src.data.data import mnist
from src.models.model import CNN

class Evaluate(object):
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

    def validation(self, model, testloader, criterion, eval = False):
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

        return (test_loss, accuracy, output, labels)

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        #parser.add_argument('load_model_from', default="")
        parser.add_argument('--ckpt', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])

        # We load the specified model based on the given args.ckpt
        #ckpt_path = "//".join(['models', args.ckpt])
        ckpt_path = args.ckpt
        #print(args.ckpt)
        model = self.load_checkpoint_CNN(ckpt_path)
        print(f"Model was loaded from {ckpt_path}.")

        # We predict on 50 images from the test set
        batch_size = 50
        train_files = ['data/processed/train_images.pt', 'data/processed/train_labels.pt']
        test_files = ['data/processed/test_images.pt', 'data/processed/test_labels.pt']
        trainloader, testloader, train_data, test_data = mnist(train_files, test_files, batch_size = batch_size)# we load all the test data
        criterion = nn.NLLLoss()

        # We carry out a forward pass with the loaded model and print the test set accuracy to the terminal
        # Turn off gradients for validation, will speed up inference
        with torch.no_grad():
            test_loss, accuracy, output, labels = self.validation(model, testloader, criterion, eval=True)

        predictions = torch.exp(output).max(1)[1]
        print("Predictions:", predictions)
        print("True labels:", labels)
        print("The accuracy on the loaded data for the loaded model is:", accuracy)

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
    Evaluate()
