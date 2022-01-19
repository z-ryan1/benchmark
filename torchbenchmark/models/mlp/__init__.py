import random

from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import numpy as np
from .mlp import MLP
from ...util.model import BenchmarkModel
from torchbenchmark.tasks import OTHER

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)


class Model(BenchmarkModel):
    task = OTHER.OTHER_TASKS

    # same hyperparameters as found on sklearn:
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    def __init__(self, device='cpu', jit=False, weight_decay=1e-4, lr=1e-3, beta_1=0.9, beta_2=0.999, eps=1e-8,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.jit = jit

        X, y = datasets.load_breast_cancer(return_X_y=True)
        X_train, X_val, y_train, y_val = map(self.array_to_tensor, train_test_split(X, y, random_state=1337))

        self.bs = min(200, X_train.shape[0])
        self.X_train = X_train[:self.bs]
        self.y_train = y_train[:self.bs]
        self.X_eval = X_val[:self.bs]

        self.model = MLP(X_train.shape[1], y_train.shape[1]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps,
                                          betas=(beta_1, beta_2))
        self.criterion = nn.BCELoss()

    def array_to_tensor(self, x):
        return torch.from_numpy(x.astype(np.float32)).reshape(-1, 1).to(self.device)

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        self.model.train()
        for _ in range(niter):
            out = self.model(self.X_train)
            self.optimizer.zero_grad()
            loss = self.criterion(out, self.y_train)
            loss.backward()
            self.optimizer.step()

    def eval(self, niter=1):
        if self.jit:
            raise NotImplementedError()
        self.model.eval()
        for _ in range(niter):
            out = self.model(self.X_eval)

    def get_module(self):
        if self.jit:
            raise NotImplementedError()
        return self.model, self.X_train


if __name__ == '__main__':
    for device in ['cpu', 'cuda']:
        print("Testing device {}, JIT {}".format(device, False))
        m = Model(device=device, jit=False)
        m.train()
        m.eval()
