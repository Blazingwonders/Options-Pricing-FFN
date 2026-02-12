import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import generate_training_data10k as gen

class NeuralNetworkPyTorch(nn.Module):

    """
    Feed-forward neural network for option pricing (call options)

    Architecture: input (dimension 10) -> 128 -> 128 -> 128 -> output layer (dimension 1)
    """

    def __init__(self, input_dim, hidden_dims):
        # output dimension is 1 (output is the options price)
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dims[0])
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.layer3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.layer4 = nn.Linear(hidden_dims[2], 1)
    
    def forward(self, x):
        a = torch.relu(self.layer1(x))
        b = torch.relu(self.layer2(a))
        c = torch.relu(self.layer3(b))
        res = self.layer4(c)
        return res
    
    




    
