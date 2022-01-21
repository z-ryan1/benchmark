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

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
