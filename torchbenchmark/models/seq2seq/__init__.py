import random

import numpy as np
import torch
import torch.nn as nn

# from ...util.model import BenchmarkModel
# from .seq2seq import DecoderRNN, EncoderRNN
from seq2seq import DecoderRNN, EncoderRNN

# from torchbenchmark.tasks import NLP

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)


# class Model(BenchmarkModel):
class Model:
    # task = NLP.TRANSLATION

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
        self.bs = batch_size
        self.device = device
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.jit = jit

        self.criterion = nn.NLLLoss()

        self.enc = EncoderRNN(input_size, hidden_size).to(device)
        self.dec = DecoderRNN(hidden_size, output_size, seq_len).to(device)

        self.encoder_optimizer = torch.optim.SGD(self.enc.parameters(), lr=lr)
        self.decoder_optimizer = torch.optim.SGD(self.dec.parameters(), lr=lr)

        # create fake data here

        input_tensor_gen = torch.Generator()
        input_tensor_gen.manual_seed(1234)

        output_tensor_gen = torch.Generator()
        output_tensor_gen.manual_seed(12345)

        self.input_tensor = torch.randint(input_size,
                                          size=(seq_len, batch_size),
                                          generator=input_tensor_gen,
                                          device=device)

        self.target_tensor = torch.randint(output_size,
                                           size=(seq_len, batch_size),
                                           generator=output_tensor_gen,
                                           device=device)

    def get_loss(self, use_teacher_forcing=False):
        init_hidden = lambda: torch.zeros(
            1, self.bs, self.hidden_size, device=self.device)

        encoder_hidden, encoder_cell = init_hidden(), init_hidden()

        encoder_outputs = torch.zeros(self.seq_len,
                                      self.bs,
                                      self.hidden_size,
                                      dtype=torch.float32,
                                      device=self.device)

        for i, input_word in enumerate(self.input_tensor):
            encoder_outputs[i], encoder_hidden, encoder_cell = self.enc(
                input_word, encoder_hidden, encoder_cell)

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        # mimic the <SOS> token
        decoder_input = torch.zeros(self.bs,
                                    dtype=torch.long,
                                    device=self.device)

        loss = 0
        for target_word in self.target_tensor:

            decoder_output, decoder_hidden, cell = self.dec(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs)

            loss += self.criterion(decoder_output, target_word)

            if use_teacher_forcing:
                decoder_input = target_word

            else:
                decoder_input = decoder_output.topk(1)[1]

        return loss

    def set_mode(self, train=True):
        self.enc.train(train)
        self.dec.train(train)

    def train(self, niter=1):
        if self.jit:
            raise NotImplementedError()

        self.set_mode(train=True)

        self.enc.train()
        self.dec.train()

        for _ in range(niter):
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()

            loss = self.get_loss(use_teacher_forcing=True)
            loss.backward()

            self.encoder_optimizer.step()
            self.decoder_optimizer.step()

    def eval(self, niter=1):
        if self.jit:
            raise NotImplementedError()

        self.set_mode(train=False)

        for _ in range(niter):
            loss = self.get_loss(use_teacher_forcing=False)

    # TODO: fix this put the entire thing in another module I guess
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
