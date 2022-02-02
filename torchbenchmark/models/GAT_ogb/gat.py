"""Code modified from 
https://github.com/pyg-team/pytorch_geometric/blob/master/examples/ogbn_products_gat.py
"""
import torch
import torch.nn.functional as F
from torch.nn import Linear as Lin
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 heads):
        super().__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(100, hidden_channels, heads))
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(heads * hidden_channels, hidden_channels, heads))
        self.convs.append(
            GATConv(heads * hidden_channels, out_channels, heads,
                    concat=False))

        self.skips = torch.nn.ModuleList()
        self.skips.append(Lin(100, hidden_channels * heads))

        for _ in range(num_layers - 2):
            self.skips.append(
                Lin(hidden_channels * heads, hidden_channels * heads))
        self.skips.append(Lin(hidden_channels * heads, out_channels))

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x, x_target), edge_index)
            x = x + self.skips[i](x_target)
            if i != self.num_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)
