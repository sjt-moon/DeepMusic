import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import math


def sliced(inputs, len_lim):
    """
    A generator that yields character slices from the training set `inputs`.
    
    :param inputs: the training set, a list of ABC tunes, with each tune starting with
           the `tune_start_symbol` ('`') and ending with the `tune_end_symbol` ('$')
    :param len_lim: a tuple (min_len, max_len), with both sides inclusive
    :yield: a random slice of input characters drawn from a random input tune
    """
    while True:
        i = inputs[np.random.randint(len(inputs))]
        l = np.random.randint(min(len(i), len_lim[0]), 1 + min(len(i), len_lim[1]))
        s = np.random.randint(0, 1 + len(i) - l)
        yield i[s:s+l]


def dataloader(dataset, len_lim, batch_size=1):
    """
    The data loader with a `sliced` iterator inside.
    
    :param dataset: the dataset
    :param len_lim: a tuple (min_len, max_len), with both sides inclusive
    :param batch_size: the batch size, default to 1
    :yield: a sequence of length `batch_size` of sequences ordered in descending order
            by their lengths
    """
    sl = sliced(dataset, len_lim)
    while True:
        seqs = [next(sl) for i in range(batch_size)]
        seqs.sort(key=len, reverse=True)
        yield seqs


def divide_train_validation(dataset):
    """
    Divide the dataset into training set and validation set in 8:2 ratio.
    
    :param dataset: the dataset to divide, with each element being a training sequence
    :return: (training set, validation set)
    """
    perm = np.random.permutation(len(dataset))
    trainset = []
    validset = []
    holdout_ratio = 0.2
    validation_start_idx = int(len(dataset) * (1 - holdout_ratio))
    for s in dataset[:validation_start_idx]:
        trainset.append(s)
    for s in dataset[validation_start_idx:]:
        validset.append(s)
    return trainset, validset


def one_hot_tensor(corp, seqs):
    """
    Make one-hot tensor, with dimension [T x B x M] where 
    - T: the length of the sequence `seq`
    - B: the batch size
    - M: the size of the corpus `corp`
    The default value of element in the tensor is zero.
    
    Reference: http://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
    
    :param corp: a list of all unique characters appearing in ABC inputs
    :param seqs: a list of input sequences, *sorted in descending order* by the sequence lengths
    :return: a FloatTensor
    """
    B = len(seqs)
    tensor = torch.zeros(reduce(max, map(len, seqs)), B, len(corp))
    for j, seq in enumerate(seqs):
        for li, letter in enumerate(seq):
            tensor[li][j][corp[letter]] = 1
    return tensor


def target_tensor(corp, seqs, remove_last_token=True):
    """
    Make the target tensor from sequences, with dimension [(T-1) x B] where
    - T: the length of the sequence `seq`; the first character of each sequence
         will be discarded
    - B: the batch size
    The default value of element in the tensor is zero.
    
    :param corp: a list of all unique characters appearing in ABC inputs
    :param seqs: a list of input sequences, *sorted in descending order* by the sequence lengths
    :return: a LongTensor
    """
    B = len(seqs)
    T = reduce(max, map(len, seqs))
    tensor = torch.zeros(T-1 if remove_last_token else T, B).long()
    for j, seq in enumerate(seqs):
        for li, letter in enumerate(seq[1:] if remove_last_token else seq):
            tensor[li][j] = corp[letter]
    return tensor


def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# Why not define an `autograd.Function` to compute the softmax with temperature?
# Since we can just divide the input by the temperature directly before feeding it
# into the original softmax. Just make sure the softmax fed into is really the
# softmax, not something called "log-softmax".





    

def predict(net_outputs, temperature=1.0, random_sample=True):
    softmax = nn.Softmax()
    net_outputs = torch.stack([softmax(net_outputs[i]) for i in range(net_outputs.size(0))]) / temperature
    if random_sample:
        samples = torch.stack([torch.stack([torch.multinomial(net_outputs[i][j], 1)
                                            for j in range(net_outputs.size(1))])
                              for i in range(net_outputs.size(0))])
    else:
        _, samples = torch.max(output, 2)
    return samples.view(net_outputs.size(0), net_outputs.size(1))


def generate(net, hidden, abc_corp, temperature=1.0, start_symbol='`', end_symbol='$', random_sample=True):
    generated = [start_symbol]
    seqs = [[ord(start_symbol)]]
    while generated[-1] != '$':
        inputs = Variable(one_hot_tensor(abc_corp, seqs))
        outputs, hidden = net(inputs, hidden)
        sample = predict(outputs, temperature=temperature, random_sample=random_sample).view(-1, 1).data[0][0]  # assume batchsize=1
        sample = abc_corp.get(sample)
        generated.append(chr(sample))
        seqs = [[sample]]
    return generated


##########################################################
#
# Trainers
#
##########################################################

def train_backprop_on_error(max_epoch, abc_corp, trainset, validset, net, cuda=False):
    train_losses = list()
    valid_losses = list()
    if cuda:
        net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters())
    start = time.time()
    
    on_error = True
    for epoch in range(max_epoch):
        
        seq = trainset[np.random.randint(len(trainset))]
        while on_error:
            hidden = net.init_hidden(cuda=cuda)
            on_error = False
            for i, ch in enumerate(seq[:-1]):
                inputs = Variable(one_hot_tensor(abc_corp, [[ch]]).cuda()) if cuda\
                         else Variable(one_hot_tensor(abc_corp, [[ch]]))
                outputs, hidden = net(inputs, hidden)
                sample = predict(abc_corp, outputs, temperature=1.0, random_sample=True).view(1).data[0]
                if sample != seq[i+1]:
                    on_error = True
                    break
            if on_error:
                targets = Variable(target_tensor(abc_corp, [[seq[i+1]]], remove_last_token=False).cuda()) if cuda\
                          else Variable(target_tensor(abc_corp, [[seq[i+1]]], remove_last_token=False))
                optimizer.zero_grad()
                loss = sum([criterion(outputs.transpose(0, 1)[i], targets.transpose(0, 1)[i])
                        for i in range(outputs.size(1))]) / outputs.size(1)
                loss.backward()
                optimizer.step()
            
            if len(train_losses) and len(train_losses) % 100 == 0:
                print '[epoch {}]'.format(epoch)
                print 'train loss: {}'.format(train_losses[-1])
                print
    return train_losses


def train_teachingforce_curriculum(max_epoch, abc_corp, trainset, validset, net, curriculum=None, cuda=False):
    def _curriculum(epoch):
        if epoch < 200:
            seqs_len = (25, 30)
        elif epoch < 400:
            seqs_len = (50, 80)
        elif epoch < 600:
            seqs_len = (200, 300)
        elif epoch < 800:
            seqs_len = (300, 500)
        else:
            seqs_len = (500, 1000)
        return seqs_len
    
    train_losses = list()
    valid_losses = list()
    
    if cuda:
        net.cuda()
    
    if curriculum is None: curriculum = _curriculum
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.004)
    prev_seqs_len = curriculum(0)  # for training
    dl = dataloader(trainset, prev_seqs_len, batch_size=1)
    start = time.time()
    for epoch in range(max_epoch):
        
        seqs_len = curriculum(epoch)
        if seqs_len != prev_seqs_len:
            dl = dataloader(trainset, seqs_len, batch_size=1)
        seqs = next(dl)
        hidden = net.init_hidden(cuda=cuda)
        if cuda:
            inputs = Variable(one_hot_tensor(abc_corp, seqs).cuda())
            targets = Variable(target_tensor(abc_corp, seqs).cuda())
        else:
            inputs = Variable(one_hot_tensor(abc_corp, seqs))
            targets = Variable(target_tensor(abc_corp, seqs))
       
        outputs, new_hidden = net(inputs, hidden)
        hidden = new_hidden
        loss = sum([criterion(outputs.transpose(0, 1)[i][:-1], targets.transpose(0, 1)[i])
                    for i in range(outputs.size(1))]) / outputs.size(1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.data[0])
        
        
        # validation
        loss = 0
        validated_char_count = 0
        for seq in validset:
            hidden = net.init_hidden(cuda=cuda)
            if cuda:
                inputs = Variable(one_hot_tensor(abc_corp, [seq]).cuda())
                targets = Variable(target_tensor(abc_corp, [seq]).cuda())
            else:
                inputs = Variable(one_hot_tensor(abc_corp, [seq]))
                targets = Variable(target_tensor(abc_corp, [seq]))
            
            outputs, new_hidden = net(inputs, hidden)
            hidden = new_hidden
            loss += sum([criterion(outputs.transpose(0, 1)[i][:-1],
                                   targets.transpose(0, 1)[i])
                        for i in range(outputs.size(1))])
            validated_char_count += outputs.size(1)
        loss /= validated_char_count
        valid_losses.append(loss.data[0])
        
        if epoch % 20 == 0:
            print '[epoch {}] {}'.format(epoch, time_since(start))
            print 'train loss:', train_losses[-1]
            print 'valid loss:', valid_losses[-1]
            print
    return train_losses, valid_losses
