import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

class RNN(nn.Module):
    '''
    Vanilla RNN implementation, with modular cell
    '''
    def __init__(self, input_size, hidden_size, cell, bias=True,
            batch_first=True):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.cell = cell
        self.num_layers = 1
        self.gate_size = self.cell.gate_size

        # Init parameters
        # Initialize weights
        self.W = Parameter(torch.Tensor(self.gate_size, input_size))
        self.U = Parameter(torch.Tensor(self.gate_size, hidden_size))

        if bias:
            self.bias_ih = Parameter(torch.Tensor(self.gate_size))
            self.bias_hh = Parameter(torch.Tensor(self.gate_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)

        self.init_weights()

    def forward(self, input, hidden):
        '''
        forward through an rnn, requires t steps
        '''

        def recurrence(input, hidden):
            return self.cell.forward(input, hidden)

        if self.batch_first:
            input = input.t()

        output = []
        steps = range(input.size(0))
        for i in steps:
            hidden = recurrence(input[i], hidden)
            # if its a cell that outputs a tuple (cell, hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
            else:
                output.append(hidden)

        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        if self.batch_first:
            output = output.t()

        return output, hidden

    def init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)


if __name__ == "__main__":
    import rnn_cells
    input_size = 1
    hidden_size = 1
    gru_cell = rnn_cells.GRUCell(input_size, hidden_size)
    gru = RNN(input_size, hidden_size, gru_cell)

    from torch.autograd import Variable as V
    input = V(torch.ones(1, 3))
    out = gru(input, input.t())
    output = out[0]
    hidden = out[1]
    print(hidden)
