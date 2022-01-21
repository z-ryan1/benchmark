import torch
import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, in_dims, out_dims, layer_size=100):
        super().__init__()
        self.hidden_layer1 = nn.Linear(in_dims, layer_size)
        self.hidden_layer2 = nn.Linear(layer_size, out_dims)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = torch.relu(x)
        x = self.hidden_layer2(x)
        x = torch.sigmoid(x)
        return x
