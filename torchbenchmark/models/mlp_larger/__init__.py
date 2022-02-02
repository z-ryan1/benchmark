import random

import numpy as np
import torch
from torch import nn
from torchbenchmark.tasks import OTHER

from ...util.model import BenchmarkModel

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)


# pretend we are using MLP to predict CIFAR images
class MLP(nn.Module):

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.main(x)


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS

    def __init__(self, device='cpu', jit=False, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.device = device
        self.jit = jit

        batch_size = 4096
        # mimic a normalized image
        self.sample_inputs = torch.randn(batch_size, 3, 32,
                                         32).clamp_(-1, 1).to(device)
        self.sample_targets = torch.randint(0, 10, (batch_size, )).to(device)

        self.model = MLP().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr,
                                          weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError()

        self.model.train()
        for _ in range(niter):
            out = self.model(self.sample_inputs)
            self.optimizer.zero_grad()
            loss = self.criterion(out, self.sample_targets)
            loss.backward()
            self.optimizer.step()

    def eval(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        self.model.eval()

        with torch.no_grad():
            for _ in range(niter):
                out = self.model(self.sample_inputs)

    def get_module(self):
        if self.jit:
            raise NotImplementedError()
        return self.model, self.sample_inputs


if __name__ == '__main__':
    for device in ['cpu', 'cuda']:
        print("Testing device {}, JIT {}".format(device, False))
        m = Model(device=device, jit=False)
        m.train()
        m.eval()
