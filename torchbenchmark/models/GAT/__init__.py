import os.path as osp

import random

import numpy as np
import torch
import torch.nn as nn

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv
from ...util.model import BenchmarkModel

from .gat import GAT

from torchbenchmark.tasks import OTHER


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS

    def __init__(self,
                 device='cpu',
                 jit=False,
                 dataset="cora",
                 lr=0.005,
                 wd=5e-4):
        super().__init__()
        self.jit = jit

        path = osp.join(osp.dirname(osp.realpath(__file__)), './', 'data',
                        dataset)
        dataset = Planetoid(path, dataset, transform=T.NormalizeFeatures())

        self.data = dataset[0].to(device)
        self.model = GAT(dataset.num_features, dataset.num_classes).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr,
                                          weight_decay=wd)

        self.criterion = nn.NLLLoss()

    def get_loss(self):
        out = self.model(self.data.x, self.data.edge_index)
        loss = self.criterion(out[self.data.train_mask],
                              self.data.y[self.data.train_mask])
        return loss

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError()

        self.model.train()

        for _ in range(niter):
            self.optimizer.zero_grad()

            loss = self.get_loss()

            loss.backward()
            self.optimizer.step()

    def eval(self, niter=1):
        if self.jit:
            raise NotImplementedError()

        self.model.eval()

        with torch.no_grad():
            for _ in range(niter):
                loss = self.get_loss()

    def get_module(self):
        if self.jit:
            raise NotImplementedError()
        return self.model, self.data


if __name__ == '__main__':
    for device in ['cpu', 'cuda']:
        print("Testing device {}, JIT {}".format(device, False))
        m = Model(device=device)
        m.train(niter=30)
        m.eval()
