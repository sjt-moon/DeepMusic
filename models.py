# minibatch=1:
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.autograd import Variable
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
#
#
# class MusicGenNet(nn.Module):
#     """
#     Architecture:
    
#         inputs [one-hot encoding]
#                   |
#                 LSTM
#                   |
#            fully-connected
    
#     Currently only support vanilla SGD optimization (no mini-batch).
#     The initial hidden states are not learnable.
#     """
    
#     def __init__(self, n_lstm, n_output,
#                  n_layers=1,
#                  using_gru=False,
#                  mini_batch_size=1):
#         """
#         The inputs are the one-hot tensor of each training sequence. The outputs
#         are the sequences of indices of characters in the corpus.
        
#         :param n_lstm: the number of LSTM units
#         :param n_output: the dimension of the output at each time step, which is
#                essentially the number of different characters in the ABC corpus
#         :param n_layers: the number of layers of LSTM, each layer having the
#                same number of units, the `n_lstm`
#         :param using_gru: True to use GRU instead of LSTM; default to False
#         :param mini_batch_size: the mini-batch size when using mini-batch SGD;
#                however, currently `mini_batch_size` must be set to one as we are
#                using vanilla SGD instead
#         """
#         nn.Module.__init__(self)
#         assert mini_batch_size == 1, 'mini_batch_size ({}) != 1'.format(mini_batch_size)
        
#         self.n_lstm = n_lstm
#         self.n_output = n_output
#         self.n_layers = n_layers
#         #self.abc_corpus = abc_corpus
#         self.mini_batch_size = mini_batch_size
        
#         self.lstm = nn.LSTM(self.n_output, self.n_lstm, num_layers=self.n_layers)
#         self.fc = nn.Linear(self.n_lstm, self.n_output)
    
#     def init_hidden(self, cuda=False):
#         hidden_size = (self.n_layers, 1, self.n_lstm)
#         if not cuda:
#             hidden = (Variable(torch.zeros(*hidden_size)),
#                       Variable(torch.zeros(*hidden_size)))
#         else:
#             hidden = (Variable(torch.zeros(*hidden_size).cuda()),
#                       Variable(torch.zeros(*hidden_size).cuda()))
#         return hidden
        
#     def forward(self, inputs, hidden):
#         """
#         :param inputs: one-hot encoding of the batch of sequences sorted in descending
#                order by sequence length; the batch size should be one
#         :param hidden: the previous hidden states
#         :return: outputs and new hidden states
#         """
#         assert inputs.size(1) == self.mini_batch_size, 'len(seqs) != self.mini_batch_size'
#         # Assuming already one-hot encoding of the inputs:
#         #inputs = trainlib.one_hot_tensor(self.abc_corpus, seqs)
        
#         # if using minibatch larger than one, this block of code must be modified,
#         # referencing: https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e
#         # begin of block
#         output, hidden = self.lstm(inputs, hidden)
#         output = self.fc(output)
#         #   current version of PyTorch only supports softmax on 2D tensor, and the softmax
#         #   is performed on the second dimension (dim=1)
#         #output = torch.stack([self.softmax(output[i]) for i in range(output.size(0))])
#         return output, hidden



import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MusicGenNet(nn.Module):
    """
    Architecture:
    
        inputs [one-hot encoding]
                  |
                LSTM
                  |
           fully-connected
    """
    
    def __init__(self, n_lstm, n_output,
                 n_layers=1,
                 using_gru=False,
                 dropout=0.0):
        """
        The inputs are the one-hot tensor of each training sequence. The outputs
        are the sequences of indices of characters in the corpus.
        
        :param n_lstm: the number of LSTM units
        :param n_output: the dimension of the output at each time step, which is
               essentially the number of different characters in the ABC corpus
        :param n_layers: the number of layers of LSTM, each layer having the
               same number of units, the `n_lstm`
        :param using_gru: True to use GRU instead of LSTM; default to False
        """
        nn.Module.__init__(self)

        self.n_lstm = n_lstm
        self.n_output = n_output
        self.n_layers = n_layers

        rnn_model = nn.GRU if using_gru else nn.LSTM
        self.rnn = rnn_model(self.n_output, self.n_lstm,
                             num_layers=self.n_layers, dropout=dropout)
        self.hidden = None  # the hidden states of `self.rnn`
        self.next_hidden = None
        self.fc = nn.Linear(self.n_lstm, self.n_output)
    
    def init_hidden(self, batch_size):
        """
        :param batch_size: the batch size
        :return: a tuple of Tensor (not Variable!)
        """
        hidden_size = (self.n_layers, batch_size, self.n_lstm)
        return (torch.zeros(*hidden_size),
                torch.zeros(*hidden_size))

    def update_hidden(self):
        """
        Replace `self.hidden` with `self.next_hidden`.
        """
        self.hidden.data = self.next_hidden.data
        
    def forward(self, inputs, seq_lens):
        """
        Assign value to `self.hidden` as wish before calling this method, but it
        must be of Variable type. Remember to repack it at least once for each
        backward propagation.

        :param inputs: one-hot encoding of the batch of sequences sorted in descending
               order by sequence length; should already be wrapped in Variable
        :param seq_lens: a list of lengths of sequences implied by `inputs`
        :return: outputs and new hidden states
        """
        packed_inputs = pack_padded_sequence(inputs, seq_lens)
        packed_outputs, self.next_hidden = self.rnn(packed_inputs, self.hidden)
        # the second return value is just `seq_lens`:
        outputs, _ = pad_packed_sequence(packed_outputs)
        outputs = outputs.transpose(0, 1)  # coordinate: [B x max(len(S)) x *]
        assert min(seq_lens) > 1, 'too small seq_lens'
        seq_lens = map(lambda x: x - 1, seq_lens)  # discard the last output of each sequence
        # coordinates: [sum(S) x *]
        outputs = torch.cat([outputs[i][:seq_lens[i]] for i in range(len(seq_lens))])
        outputs = self.fc(outputs)
        #   current version of PyTorch only supports softmax on 2D tensor, and the softmax
        #   is performed on the second dimension (dim=1)
        #output = torch.stack([self.softmax(output[i]) for i in range(output.size(0))])
        return outputs
