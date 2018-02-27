import torch
import torch.nn as nn
from torch.autograd import Variable

class Music(nn.Module):
    def __init__(self, voc_size, embedding_dim=100, hidden_size=50 ,num_layers=1, use_gpu=True):
        super(Music, self).__init__()
        self.voc_size = voc_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # (|Vocabulary|, embedding_dim)
        self.embedding = nn.Embedding(voc_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers)
        self.decoder = nn.Linear(hidden_size, voc_size)

        #self.hidden = self.init_hidden()
        self.use_gpu = use_gpu

    def forward(self, input, hidden):
        '''
        
        @para:
        input: (# of chars, embedding dim)
        hidden: hidden state & cell state
        '''
        input = self.embedding(input.view(1, -1))
        output, hidden = self.lstm(input.view(1,1,-1), hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        h = torch.randn(self.num_layers, 1, self.hidden_size)
        c = torch.randn(self.num_layers, 1, self.hidden_size)
        if self.use_gpu:
            h = h.cuda()
            c = c.cuda()
        h = Variable(h)
        c = Variable(c)
        return (h, c)
