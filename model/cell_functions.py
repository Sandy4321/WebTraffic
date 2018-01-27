import torch.nn.functional as F


def lstm(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    '''
    LSTM cell math works.
    Assumes weight matrix (4 x hidden_size) of the following format:
        - 1 column - forget_gate
        - 2 column - in_gate
        - 3 column - cell_gate
        - 4 column - out_gate
    '''
    gates_inp = F.linear(input, w_ih, b_ih)
    gates_hid = F.linear(hidden, w_hh, b_hh)
    gates = gates_inp + gates_hid
    forget, inp, cell, out = gates.chunk(4, 1)

    forget = F.sigmoid(forget)
    inp = F.sigmoid(inp)
    cell = F.tanh(cell)
    out = F.sigmoid(out)

    cell_state = hidden * forget + (inp * cell)    
    hidden_state = out * F.tanh(cell_state)

    return (hidden_state, cell_state)


def gru(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    '''
    GRU cell quickmaths.

    Assumes weight matrix(3 x hidden_size) of the following format:
        - 1 column - reset_gate
        - 2 column - update_gate
        - 3 column - out_gate
    '''

    gates_inp = F.linear(input, w_ih, b_ih)
    gates_hid = F.linear(hidden, w_ih, b_ih)
    gates = gates_inp + gates_hid

    reset, update, out = gates.chunk(3, 1)

    reset_gate = F.sigmoid(reset)
    update_gate = F.sigmoid(update)
    out_gate = F.tanh(reset_gate + out)

    #print(reset_gate.shape)
    #print(update_gate.shape)
    #print(out_gate.shape)

    one_update = (1 - update_gate)
    #print(one_update)
    hidden_update = one_update * hidden
    hidden_state = hidden_update + (update_gate * out_gate)

    return hidden_state


def rnn_tanh(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    '''
    Simple RNN tanh cell
    '''
    hidden_state = F.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh,
        b_hh))
    return hidden_state


def rnn_relu(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    '''
    Simple RNN ReLU cell
    '''
    hidden_state = F.relu(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh,
        b_hh))
    return hidden_state
