import os.path as osp

import random

import numpy as np
import torch
import torch.nn as nn

from ...util.model import BenchmarkModel

from .gat import GAT

from torchbenchmark.tasks import OTHER

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS

    def __init__(self, device='cpu', jit=False):
        super().__init__()
        self.jit = jit

        num_classes = 47
        num_features = 100
        num_samples = 2449029

        self.batch_size = 512

        meta_path = osp.join(osp.dirname(osp.realpath(__file__)), 'meta.pt')
        meta_dict = torch.load(meta_path)
        self.adjs = [x.to(device) for x in meta_dict['adjs']]
        self.n_id = meta_dict['n_id'].to(device)

        self.x = torch.randn(num_samples,
                             num_features,
                             device=device,
                             dtype=torch.float32)
        self.y = torch.randint(0,
                               num_classes, (num_samples, ),
                               device=device,
                               dtype=torch.long)

        self.model = GAT(num_features, 128, num_classes, num_layers=3,
                         heads=4).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.criterion = nn.NLLLoss()

    def get_loss(self):
        out = self.model(self.x[self.n_id], self.adjs)
        loss = self.criterion(out, self.y[self.n_id[:self.batch_size]])
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
        return self.model, (self.x, self.y)


if __name__ == '__main__':
    for device in ['cpu', 'cuda']:
        print("Testing device {}, JIT {}".format(device, False))
        m = Model(device=device)
        m.train(niter=30)
        m.eval()
