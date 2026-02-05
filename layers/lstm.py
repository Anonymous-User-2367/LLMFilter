import torch
import torch.nn as nn

class LSTM(nn.Module):
    '''
    LSTM model is composed of three parts: a word embedding layer, an LSTM network, and an output layer.
    The word embedding layer takes a sequence of word indices (from the vocabulary) as input 
    and outputs a sequence of vectors, each representing a word embedding.
    The LSTM network takes each word embedding as input and outputs a hidden feature for each word embedding.
    The output layer takes the hidden feature and outputs the probability of each word in the vocabulary.
    Feel free to modify the init arguments if necessary.
    '''
    def __init__(self, 
                 f_in, 
                 f_out, 
                 hidden_dim=256, 
                 hidden_layers=2, 
                 dropout=0.1,
                 activation='tanh'):
        super(LSTM, self).__init__()
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
            raise NotImplementedError("Unsupported activation function")

        # LSTM layer
        self.lstm = nn.LSTM(f_in, hidden_dim, hidden_layers, dropout=dropout, batch_first=True)
        
        # Fully connected decoder
        self.decoder = nn.Linear(hidden_dim, f_out)
        self.init_weights()

    def init_weights(self):
        init_uniform = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input):
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(input)
        
        # Apply dropout
        output = self.drop(output)
        
        # Decode the output to vocabulary probabilities
        decoded = self.decoder(output)
        
        return decoded
