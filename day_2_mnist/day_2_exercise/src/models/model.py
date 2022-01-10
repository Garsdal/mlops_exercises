from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class FFNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        ''' Builds a feedforward network with arbitrary hidden layers.
        
            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        
        '''
        super().__init__()
        # Input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        
        for each in self.hidden_layers:
            x = F.relu(each(x))
            x = self.dropout(x)
        x = self.output(x)
        
        return F.log_softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self, num_classes, channels, height, width, num_filters):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.height = height
        self.width = width
        self.num_filters = num_filters
        
        # Not good code but we save parameters manually
        num_filters_conv1 = num_filters
        kernel_size_conv1 = 5 # [height, width]
        stride_conv1 = 1 # [stride_height, stride_width]
        stride_conv2 = 1
        num_l1 = 100
        padding_conv1 = 2

        # Parameters for maxpooling    
        kernel_size_maxpool = 2
        stride_maxpool = 2
        padding_maxpool = 0
        dilation_maxpool = 0

        # First layer
        self.conv_1 = nn.Conv2d(in_channels=channels,
                            out_channels=num_filters_conv1,
                            kernel_size=kernel_size_conv1,
                            stride=stride_conv1,
                            padding = padding_conv1)
        
        self.conv_out_height = self.compute_conv_dim(self.height, kernel_size_conv1, padding_conv1, stride_conv1)
        self.conv_out_width = self.compute_conv_dim(self.width, kernel_size_conv1, padding_conv1, stride_conv1)
        
        # Second layer
        self.conv_2 = nn.Conv2d(in_channels=num_filters_conv1,
                            out_channels=num_filters_conv1,
                            kernel_size=kernel_size_conv1,
                            stride=stride_conv1,
                            padding = padding_conv1)
        
        self.conv_out_height2 = self.compute_conv_dim(self.conv_out_height, kernel_size_conv1, padding_conv1, stride_conv1)
        self.conv_out_width2 = self.compute_conv_dim(self.conv_out_width, kernel_size_conv1, padding_conv1, stride_conv1)
        
        # Max pooling
        self.maxpool1 = nn.MaxPool2d(kernel_size = kernel_size_maxpool,
                      stride = stride_maxpool)
        
        self.maxpool_out_height = self.compute_maxpool_dim(self.conv_out_height2, kernel_size_maxpool, padding_maxpool, stride_maxpool, dilation_maxpool)
        self.maxpool_out_width = self.compute_maxpool_dim(self.conv_out_width2, kernel_size_maxpool, padding_maxpool, stride_maxpool, dilation_maxpool)
        
        # Linear output
        self.l1_in_features = int((num_filters_conv1 * self.maxpool_out_height * self.maxpool_out_width))
        
        self.l_1 = nn.Linear(in_features=self.l1_in_features, 
                          out_features=num_l1,
                          bias=True)
        
        # Output layer with num_classes
        self.l_out = nn.Linear(in_features=num_l1, 
                            out_features=num_classes,
                            bias=False)
        
    def forward(self, x):
        
        # Convolutional layer
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        # Max pooling
        x = self.maxpool1(x)
        
        # Unwrapping to linear layers
        x = x.view(-1, self.l1_in_features)
        
        # Output layer
        x = F.relu(self.l_1(x))
        
        return F.softmax(self.l_out(x), dim=1)

    # Also assuming that the kernel is quadratic
    def compute_conv_dim(self,dim_size, kernel_size_conv1, padding_conv1, stride_conv1):
        return int((dim_size - kernel_size_conv1 + 2 * padding_conv1) / stride_conv1 + 1)

    # This is assuming that the kernel is quadratic
    def compute_maxpool_dim(self,dim_size, kernel_size_maxpool, padding_maxpool, stride_maxpool, dilation_maxpool):
        return int( (dim_size - dilation_maxpool*(kernel_size_maxpool-1) + 2 * padding_maxpool - 1) / stride_maxpool + 1)
