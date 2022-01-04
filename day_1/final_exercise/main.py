import argparse
import sys

import torch
from torch import nn, optim

from data import mnist
from model import FFNN, CNN

import matplotlib.pyplot as plt
import numpy as np

class TrainOREvaluate(object):
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

        # I am not sure about the above code so i try to have the command here
        #self.action = args.command
    
    def train(self, optimizer=None, epochs=5, print_every=64):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        parser.add_argument('--epochs', default=5)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here

        # FFNN
        #n_obs = 784
        #n_out = 10
        #n_hid1 = 512
        #n_hid2 = 256
        #n_hid3 = 128
        #model = FFNN(n_obs, n_out, [n_hid1, n_hid2, n_hid3])
        #criterion = nn.NLLLoss()
        #optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        #print(model)

        batch_size = 64
        trainloader, testloader = mnist(batch_size)
        images, labels = next(iter(trainloader))

        # Reshape an image to the correct size
        images = images.view(batch_size, 1, images.shape[1], images.shape[2])

        # hyperameters of the model
        num_classes = 10
        channels = images.shape[1]
        height = images.shape[2]
        width = images.shape[3]

        # Define model
        model = CNN(num_classes, channels, height, width)
        print(model)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # SHOULD BE DYNAMIC TOO
        epochs = 3
        steps = 0
        running_loss = 0

        train_loss = []
        for e in range(epochs):
            # Model in training mode, dropout is on
            model.train()
            for images, labels in trainloader:
                steps += 1
                
                # FFNN
                # Flatten images into a 784 long vector
                #images.resize_(images.size()[0], 784)

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
                        test_loss, accuracy = self.validation(model, testloader, criterion)
                    
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                        "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))
                    
                    running_loss = 0
                    
                    # Make sure dropout and grads are on for training
                    model.train()

        # Finally we produce a single plot with the training curve
        with torch.no_grad():
            plt.figure()
            plt.plot(train_loss, 'g', label='Training loss', zorder=1)
            plt.legend()
            plt.savefig('training_curve.png', dpi=500)

        # We save the model at | SHOULD BE DYNAMIC:
        print("Model was saved as 'checkpoint.pth'.")

        # FFNN
        # checkpoint = {'input_size': 784,
        #               'output_size': 10,
        #               'hidden_layers': [each.out_features for each in model.hidden_layers],
        #               'state_dict': model.state_dict()}

        # CNN
        checkpoint = {'num_classes': num_classes,
                        'channels': channels,
                        'height': height,
                        'width': width,
                        'state_dict': model.state_dict()}

        torch.save(checkpoint, 'checkpoint_CNN.pth')

    def validation(self, model, testloader, criterion, eval = False):
        accuracy = 0
        test_loss = 0
        for images, labels in testloader:
            
            # FFNN
            #images = images.resize_(images.size()[0], 784)

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

    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here | We load our CNN model
        print(args.load_model_from)
        model = self.load_checkpoint_CNN(args.load_model_from)
        print("Model was loaded from 'checkpoint.pth'.")

        batch_size = 5000
        trainloader, testloader = mnist(batch_size)
        criterion = nn.NLLLoss()

        # We carry out a forward pass with the loaded model and print the test set accuracy to the terminal
        # Turn off gradients for validation, will speed up inference
        with torch.no_grad():
            test_loss, accuracy = self.validation(model, testloader, criterion, eval=True)

        print("Test accuracy:", accuracy)

    def load_checkpoint_FFNN(self, filepath):
        checkpoint = torch.load(filepath)
        model = FFNN(checkpoint['input_size'],
                                checkpoint['output_size'],
                                checkpoint['hidden_layers'])
        model.load_state_dict(checkpoint['state_dict'])
        
        return model

    def load_checkpoint_CNN(self, filepath):
        checkpoint = torch.load(filepath)
        model = CNN(checkpoint['num_classes'],
                                checkpoint['channels'],
                                checkpoint['height'],
                                checkpoint['width'])
        model.load_state_dict(checkpoint['state_dict'])
        
        return model

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    


    
    
    
    
    