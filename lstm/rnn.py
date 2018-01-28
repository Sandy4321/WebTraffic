import math

import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.optim as optim

import rnn_cells
from torch.autograd import Variable as V


class RNN(nn.Module):
    '''
    Vanilla RNN implementation, with modular cell
    '''
    def __init__(self, input_size, hidden_size, cell, bias=True,
            first_batch=False):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = first_batch
        self.cell = cell
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

    def forward(self, input, hidden=None, future=0):
        '''
        forward through an rnn, requires t steps
        '''
        print(input.shape)
        if hidden is None:
            hidden = Variable(torch.zeros(self.hidden_size, input.size(1))).double()

        # asserts at cell level for the whole input
        #self.cell.assert_forward(input)
        #self.cell.assert_hidden(input, hidden)

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

        for i in range(future):
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
    #input_size = 1
    #hidden_size = 1
    #gru_cell = rnn_cells.GRUCell(input_size, hidden_size)
    #gru = RNN(input_size, hidden_size, gru_cell)

    #input = V(torch.ones(1, 3))
    #out = gru(input, input.t())
    #output = out[0]
    #hidden = out[1]
    #print(hidden)

    # --------------------------------------------------------------------------
    np.random.seed(2)

    T = 20
    L = 1000
    N = 100

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')

    # --------------------------------------------------------------------------
    input = V(torch.from_numpy(data[3:, :-1]), requires_grad=False)
    target = V(torch.from_numpy(data[3:, 1:]), requires_grad=False)
    test_input = V(torch.from_numpy(data[:3, :-1]), requires_grad=False)
    test_target = V(torch.from_numpy(data[:3, 1:]), requires_grad=False)

    # --------------------------------------------------------------------------
    def train():
        np.random.seed(0)
        torch.manual_seed(0)

        input_size = 999
        hidden_size = 999
        gru_cell = rnn_cells.GRUCell(input_size, hidden_size, bias=False)
        gru = RNN(input_size, hidden_size, gru_cell, bias=False)
        gru.double()
        criterion = nn.MSELoss()

        optimizer = optim.LBFGS(gru.parameters(), lr=0.8)

        hidden_state = []
        def train_step():
            optimizer.zero_grad()
            out = gru(input)
            loss = criterion(out[0], target)
            hidden_state.append(out[1])
            print('loss: ', loss.data.numpy()[0])
            loss.backward()
            return loss

        for i in range(15):
            print("Step: %d" % i) 
            optimizer.step(train_step)

            # PREDICTIONS
            future = 1000
            pred = gru(test_input, hidden_state[-1], future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss: ', loss.data.numpy()[0])

            y = pred.data.numpy()

    train()
