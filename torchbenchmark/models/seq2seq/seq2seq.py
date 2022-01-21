import torch
import torch.nn as nn

"""Based on the tutorial
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
"""


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    def forward(self, x, hidden=None, cell=None):
        embedded = self.embedding(x).unsqueeze(0)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return output, hidden, cell


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, seq_len):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.attn = nn.Sequential(
            nn.Linear(2 * hidden_size, seq_len),
            nn.Softmax(dim=-1),
        )

        self.attn_combine = nn.Sequential(
            nn.Linear(2 * hidden_size, self.hidden_size),
            nn.ReLU(),
        )

    def forward(self, x, hidden, cell, encoder_outputs):
        x = self.embedding(x).unsqueeze(0).relu()
        attn_weights = self.attn(torch.dstack((x, hidden)))

        attn_values = torch.einsum("lbs, sbd -> lbd", attn_weights,
                                   encoder_outputs)

        new_input = self.attn_combine(torch.dstack((x, attn_values)))

        output, (hidden, cell) = self.lstm(new_input, (hidden, cell))
        output = self.softmax(self.out(output[0]))
        return output, hidden, cell


class Seq2Seq(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, batch_size,
                 seq_len):
        super().__init__()
        self.enc = EncoderRNN(input_size, hidden_size)
        self.dec = DecoderRNN(hidden_size, output_size, seq_len)
        self.criterion = nn.NLLLoss()

        self.seq_len = seq_len
        self.bs = batch_size
        self.hidden_size = hidden_size

    def init_hidden(self, device):
        return torch.zeros(1, self.bs, self.hidden_size, device=device)

    def forward(self, input_tensor, target_tensor, use_teacher_forcing):
        device = input_tensor.device

        encoder_hidden, encoder_cell = self.init_hidden(
            device), self.init_hidden(device)

        encoder_outputs = torch.zeros(self.seq_len,
                                      self.bs,
                                      self.hidden_size,
                                      dtype=torch.float32,
                                      device=device)

        for i, input_word in enumerate(input_tensor):
            encoder_outputs[i], encoder_hidden, encoder_cell = self.enc(
                input_word, encoder_hidden, encoder_cell)

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        # mimic the <SOS> token
        decoder_input = torch.zeros(self.bs, dtype=torch.long, device=device)

        loss = 0
        for target_word in target_tensor:

            decoder_output, decoder_hidden, cell = self.dec(
                decoder_input, decoder_hidden, decoder_cell, encoder_outputs)

            loss += self.criterion(decoder_output, target_word)

            if use_teacher_forcing:
                decoder_input = target_word

            else:
                decoder_input = decoder_output.topk(1)[1].flatten()

        return loss
