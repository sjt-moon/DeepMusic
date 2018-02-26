from os import path
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import OneHotEncoder

class Trainer():
    def __init__(self, model, char2idx_dict, idx2char_dict, chunk_size=25, lr=0.001):
        self.chunk_size = chunk_size
        self.model = model
        self.char2idx_dict = char2idx_dict
        self.idx2char_dict = idx2char_dict
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def random_chunk(self, data):
        '''Get a chunk of chars randomly.

        @para:
        data: chars of training data    
        '''
        start_idx = random.randint(0, len(data) - self.chunk_size)
        return data[start_idx: start_idx + self.chunk_size + 1]

    def char2idx(self, seq):
        tensor = torch.zeros(len(seq)).long()
        for i,c in enumerate(seq):
            tensor[i] = self.char2idx_dict[c]
        return Variable(tensor)

    def get_next_batch(self, data):
        chunk = self.random_chunk(data)
        input = self.char2idx(chunk[:-1])
        teacher = self.char2idx(chunk[1:])
        return input, teacher

    def _fit(self, input, teacher):
        '''fit
        
        @para
        input: LongTensor, idx
        teacher: LongTensor, idx
        '''
        hidden = self.model.init_hidden()
        self.model.zero_grad()
        loss = 0.0

        for i in range(self.chunk_size):
            output, hidden = self.model(input[i], hidden)
            loss += self.criterion(output, teacher[i])
        loss.backward()
        self.optimizer.step()
        return loss.data[0] / self.chunk_size

    def fit(self, data, max_iter=2000, log_freq=100):
        losses = []
        avg_loss = 0
        for epoch in range(1, max_iter+1):
            loss = self._fit(*self.get_next_batch(data))
            avg_loss += loss

            if epoch % log_freq == 0:
                print('epoch %d, loss %.3f' % (epoch, loss))
                avg_loss /= log_freq
                losses.append(avg_loss)
                avg_loss = 0.0
        return losses

    def predict(self, data):
        '''predict sequence
        
        @para
        data: list of chars
        '''
        input = self.char2idx(data)
        hidden = self.model.init_hidden()
        for c in input:
            output, hidden = self.model(c, hidden)
        # to be continue

    def inference(self, start_tune='<start>', size=200, temp=0.6):
        '''Generate tunes.'''
        output_tune = ''

        hidden = self.model.init_hidden()
        #start_tune = [c for c in start_tune]
        start_tune_idx = self.char2idx(start_tune)

        for i in range(len(start_tune)-1):
            _, hidden = self.model(start_tune_idx[i], hidden)

        input = start_tune_idx[-1]
        for i in range(size):
            output, hidden = self.model(input, hidden)

            # sampling
            output_dist = output.data.view(-1).div(temp).exp()
            sample_idx = torch.multinomial(output_dist, 1)[0]
            sample = self.idx2char_dict[sample_idx]
            output_tune += sample
            input = self.char2idx(sample)
        return output_tune

def get_dicts(data):
    char2idx_dict, idx2char_dict = {}, {}
    voc = set(data)
    for i,c in enumerate(voc):
        char2idx_dict[c] = i
        idx2char_dict[i] = c
    return char2idx_dict, idx2char_dict

def get_data(filename='../data/input.txt'):
    assert path.isfile(filename)
    return [d for d in (open(filename)).read()]

def split_data(data, ratios=[0.8, 0.2]):
    size = int(ratios[0] * len(data))
    return data[:size], data[size:]

# this is almost useless
def get_dateset(filename='../data/input.txt', split_ratio=[0.8,0.2], batch_size=25, use_one_hot=True):
    '''Maybe I should split dataset by <start> and <end> into sequences?'''

    assert path.isfile(filename)
    assert np.sum(split_ratio) == 1

    data = [d for d in (open(filename)).read()]
    vocabulary = sorted(set(data))
    char2idx, idx2char = {}, {}
    for i,c in enumerate(vocabulary):
        char2idx[c] = i
        idx2char[i] = c

    # only record indices instead of chars
    data = np.array([char2idx[c] for c in data])
    train_size = int(split_ratio[0] * len(data))

    # col vector x_train
    x_train = np.reshape(data[:train_size], (-1,1))
    #y_train = data[1:train_size+1]

    x_valid = np.reshape(data[train_size:-1], (-1,1))
    #y_valid = data[train_size+1:]

    dimensionality = 1
    
    # y is still NOT one-hot encoded
    if use_one_hot:
        enc = OneHotEncoder()
        x_train = enc.fit_transform(x_train).toarray()
        x_valid = enc.transform(x_valid).toarray()
        dimensionality = len(x_train[0])

    # toTensor
    train_size = int(len(x_train) / batch_size) * batch_size
    x_train = x_train[0:train_size, :]
    # shape: (# of mini batches, batch size, one hot dimensionality)
    x_train = np.reshape(x_train, (int(train_size/batch_size), batch_size, dimensionality))
    y_train = x_train[:, 1:, :]
    x_train = x_train[:, :-1, :]

    valid_size = int(len(x_valid) / batch_size) * batch_size
    x_valid = x_valid[0:valid_size, :]
    x_valid = np.reshape(x_valid, (int(valid_size/batch_size), batch_size, dimensionality))
    y_valid = x_valid[:, 1:, :]
    x_valid = x_valid[:, :-1, :]

    return x_train, y_train, x_valid, y_valid


    


