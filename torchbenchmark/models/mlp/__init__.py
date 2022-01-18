import random

from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
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
        if self.device == 'cuda':
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = False

        X, y = datasets.load_breast_cancer(return_X_y=True)
        X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=1337)
        X_train, X_val, y_train, y_val = X_train.astype(np.float32), X_val.astype(np.float32), \
                                         y_train.astype(np.float32), y_val.astype(np.float32)
        X_train, X_val, y_train, y_val = torch.from_numpy(X_train), torch.from_numpy(X_val), \
                                         torch.from_numpy(y_train), torch.from_numpy(y_val)
        y_train = y_train.reshape(-1, 1)
        y_val = y_val.reshape(-1, 1)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_val, y_val)
        self.bs = min(200, X_train.shape[0])
        self.trainloader = DataLoader(train_dataset, batch_size=self.bs)
        self.testloader = DataLoader(test_dataset, batch_size=self.bs)
        self.X_train, self.y_train = next(iter(self.trainloader))
        self.X_eval, self.y_eval = next(iter(self.testloader))
        self.X_train, self.y_train = self.X_train.to(self.device), self.y_train.to(self.device)
        self.X_eval, self.y_eval = self.X_eval.to(self.device), self.y_eval.to(self.device)

        self.model = MLP(X_train.shape[1], y_train.shape[1]).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay, eps=eps,
                                          betas=(beta_1, beta_2))
        self.criterion = nn.BCELoss()

    def train(self, niter=1):
        self.model.train()
        for _ in range(niter):
            out = self.model(self.X_train)
            self.optimizer.zero_grad()
            loss = self.criterion(out, self.y_train)
            loss.backward()
            self.optimizer.step()

    def eval(self, niter=1):
        self.model.eval()
        for _ in range(niter):
            out = self.model(self.X_eval)

    def get_module(self):
        return self.model, self.X_train


if __name__ == '__main__':
    for device in ['cpu', 'cuda']:
        print("Testing device {}, JIT {}".format(device, 'False'))
        m = Model(device=device, jit=False)
        m.train()
        m.eval()
