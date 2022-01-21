import random

import numpy as np
import torch
import torch.nn as nn
from torchbenchmark.tasks import NLP

from ...util.model import BenchmarkModel
from .seq2seq import Seq2Seq


class Model(BenchmarkModel):
    task = NLP.TRANSLATION

    def __init__(self,
                 device='cpu',
                 jit=False,
                 lr=0.01,
                 input_size=5000,
                 output_size=5000,
                 hidden_size=512,
                 batch_size=256,
                 seq_len=100):
        super().__init__()

        self.model = Seq2Seq(input_size, output_size, hidden_size, batch_size,
                             seq_len).to(device)
        self.jit = jit

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        # create fake data here

        input_tensor_gen = torch.Generator()
        input_tensor_gen.manual_seed(1234)

        output_tensor_gen = torch.Generator()
        output_tensor_gen.manual_seed(12345)

        self.input_tensor = torch.randint(
            input_size,
            size=(seq_len, batch_size),
            generator=input_tensor_gen,
        ).to(device)

        self.target_tensor = torch.randint(
            output_size,
            size=(seq_len, batch_size),
            generator=output_tensor_gen,
        ).to(device)

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError()

        self.model.train()

        for _ in range(niter):
            self.optimizer.zero_grad()

            loss = self.model(self.input_tensor,
                              self.target_tensor,
                              use_teacher_forcing=True)
            loss.backward()

            self.optimizer.step()

    def eval(self, niter=1):
        if self.jit:
            raise NotImplementedError()

        self.model.eval()

        with torch.no_grad():
            for _ in range(niter):
                loss = self.model(self.input_tensor,
                                  self.target_tensor,
                                  use_teacher_forcing=False)

    def get_module(self):
        if self.jit:
            raise NotImplementedError()
        return self.model, (self.input_tensor, self.target_tensor)


if __name__ == '__main__':
    for device in ['cpu', 'cuda']:
        print("Testing device {}, JIT {}".format(device, False))
        m = Model(device=device,
                  jit=False,
                  input_size=10,
                  output_size=10,
                  hidden_size=128,
                  batch_size=128,
                  seq_len=32)
        m.train(niter=30)
        m.eval()
