import torch.nn as nn

class RNN(nn.Module):
    '''
    RNN model is composed of three parts: a word embedding layer, a rnn network and a output layer
    The word embedding layer have input as a sequence of word index (in the vocabulary) 
    and output a sequence of vector where each one is a word embedding
    The network has input of each word embedding and output a hidden feature corresponding to each word embedding
    The output layer has input as the hidden feature and output the probability of each word in the vocabulary
    feel free to change the init arguments if necessary
    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=256, 
                 hidden_layers=2, 
                 dropout=0.1,
                 activation='tanh'): 
        super(RNN, self).__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.drop = nn.Dropout(dropout)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise NotImplementedError

        self.gru = nn.GRU(f_in, hidden_dim, hidden_layers, dropout=dropout)
        self.decoder = nn.Linear(hidden_dim, f_out)
        self.init_weights()
    
    def init_weights(self):
        init_uniform = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        output, hidden = self.gru(input)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded


