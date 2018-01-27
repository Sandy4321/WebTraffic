import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from cell_functions import lstm, gru, rnn_tanh, rnn_relu


class CellBase(nn.Module):
    def assert_forward(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                    'input_size 1: {}, self input size: {}'.format(
                            input.size(1), self.input_size
                        )

                    )

    def assert_hidden(self, input, hidden):
        if input.size(0) != hidden.size(0):
            raise RuntimeError(
                    'input_size 0: {}, hidden size 0: {}'.format(
                        input.size(0), hidden.size(0)))

        if hidden.size(1) != self.hidden_size:
            raise RuntimeError(
                    'hidden_size 1: {}, self hidden size: {}'.format(
                        hidden.size(1), self.hidden_size))


class RNNCell(CellBase):
    '''
    RNN Cell with a specific nonlinearity function
    '''
    def __init__(self, input_size, hidden_size, cell_func, bias=True):
        super(RNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.gate_size = hidden_size

        # TODO assert function good behavior
        self.cell = cell_func
        
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
        # TODO assert input, and hidden dimensionality
        output = self.cell(
                input, hidden,
                self.W, self.U,
                self.bias_ih, self.bias_hh
                )
        return output

    def init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)


class TanhCell(RNNCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(TanhCell, self).__init__(input_size, hidden_size, rnn_tanh)


class ReluCell(RNNCell):
    def __init__(self, input_size, hidden_size, bias=True):
        super(ReluCell, self).__init__(input_size, hidden_size, rnn_relu)


class LSTMCell(CellBase):
    '''
    RNN Cell with a specific nonlinearity function
    '''
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.gate_size = 4 * hidden_size

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
        # TODO assert input, and hidden dimensionality
        output = lstm(
                input, hidden,
                self.W, self.U,
                self.bias_ih, self.bias_hh
                )
        return output

    def init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)


class GRUCell(CellBase):
    '''
    RNN Cell with a specific nonlinearity function
    '''
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.gate_size = 3 * hidden_size

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
        # TODO assert input, and hidden dimensionality
        output = gru(
                input, hidden,
                self.W, self.U,
                self.bias_ih, self.bias_hh
                )
        return output

    def init_weights(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
