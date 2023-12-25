import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    '''
    params:
        vocal_length: int, the length of vocal
        hidden_size: int, the size of hidden layer
        n_layers: int, the number of layers of rnn
    '''
    def __init__(self, vocal_length, hidden_size, n_layers=2, device='cuda'):
        super(Encoder, self).__init__()
        self.input_size = vocal_length
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.device = torch.device(device)
        
        self.embedding = nn.Embedding(self.input_size, hidden_size)
        self.rnn = nn.GRU(self.hidden_size, hidden_size)
    
    def forward(self, input, hidden=None):
        if hidden is None:
            # Then init the hidden state
            hidden = torch.zeros(1, 1, self.hidden_size).to(self.device)
        
        embedded = self.embedding(input).view(1, 1, -1)
        # Enc
        for _ in range(self.n_layers):
            output, hidden = self.rnn(embedded, hidden)
        
        return output, hidden


class Decoder(nn.Module):
    '''
    params:
        input_size: encoder's hidden_size
        vocal_length: int, the length of vocal, also the output_size
        n_layers: int, the number of layers of rnn
    '''
    def __init__(self, input_size, vocal_length, n_layers=2, device='cuda'):
        self.input_size = input_size
        self.output_size = vocal_length
        self.n_layers = n_layers
        self.device = torch.device(device)
        
        self.embedding = nn.Embedding(self.output_size, self.input_size)
        self.rnn = nn.GRU(self.input_size, self.input_size)
        
    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = torch.zeros(1, 1, self.input_size).to(self.device)
        
        for _ in range(self.n_layers):
            output, hidden = self.rnn(input, hidden)
        
        output = self.embedding(output[0])
        output = F.log_softmax(output, dim=1)
        
        return output, hidden
    
    
class AttentionDecoder(nn.Module):
    def __init__(self, input_size, vocal_length, n_layers=2, device='cuda'):
        self.input_size = input_size
        self.output_size = vocal_length
        self.n_layers = n_layers
        self.device = torch.device(device)
        
        self.embedding = nn.Embedding(self.output_size, self.input_size)
        self.rnn = nn.GRU(self.input_size, self.input_size)
    
    def forward(self, input, hidden=None):
        if hidden is None:
            hidden = torch.zeros(1, 1, self.input_size).to(self.device)
        
        for _ in range(self.n_layers):
            output, hidden = self.rnn(input, hidden)
        
        output = self.embedding(output[0])
        output = F.log_softmax(output, dim=1)
        
        return output, hidden


class Seq2Seq(nn.Module):
    pass