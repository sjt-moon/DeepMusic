import time
import datetime
import trainlib
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import itertools
import pickle


def prepare_inputs(corpus, seqs, cuda=False, requires_grad=False):
    inputs = trainlib.one_hot_tensor(corpus, seqs)
    if cuda:
        inputs = Variable(inputs.cuda(), requires_grad=requires_grad)
    else:
        inputs = Variable(inputs, requires_grad=requires_grad)
    return inputs


class Trainer:
    def __init__(self, net, loaders, corpus, criterion, optimizer,
                 cuda, log_period=True, gen_period=True):
        self.net = net
        self.loaders = loaders  # {'train': train_loader, 'valid': valid_loader}
        self.corpus = corpus
        self.criterion = criterion
        self.optimizer = optimizer
        self.cuda = cuda

        # print stat every `log_period` epochs;
        # False to disable printing
        # True to print every 20 epochs
        if log_period is False:
            self.log_period = None
        elif log_period is True:
            self.log_period = 20
        else:
            self.log_period = log_period
        
        if gen_period is False:
            self.gen_period = None
        elif gen_period is True:
            self.gen_period = 100
        else:
            self.gen_period = gen_period

        self.epoch = 0  # accumulated epoch
        self.train_losses = list()
        self.valid_losses = list()
        self.samples = dict()  # { epoch: samples }
        
        self.softmax = nn.Softmax()

    def prepare_inputs_targets(self, seqs, requires_grad=False):
        """
        :param seqs: sequence of training sequences
        :return: inputs (Variable), targets (Variable)
        """
        inputs = trainlib.one_hot_tensor(self.corpus, seqs)
        if self.cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs, requires_grad=requires_grad)

        target_seqs = map(lambda s: s[1:], seqs)
        target_seqs = list(itertools.chain.from_iterable(target_seqs))
        # transform vocabularies to indices:
        target_seqs = map(lambda v: self.corpus[v], target_seqs)
        targets = torch.LongTensor(target_seqs)
        if self.cuda:
            targets = targets.cuda()
        targets = Variable(targets, requires_grad)

        return inputs, targets

    def prepare_hidden(self, batch_size, requires_grad=False):
        hidden = self.net.init_hidden(batch_size)
        if type(self.net.rnn) is nn.LSTM:
            if self.cuda:
                hidden = (hidden[0].cuda(), hidden[1].cuda())
            hidden = (Variable(hidden[0], requires_grad=requires_grad),
                      Variable(hidden[1], requires_grad=requires_grad))
        else:
            if self.cuda:
                hidden = hidden.cuda()
            hidden = Variable(hidden, requires_grad=requires_grad)
        return hidden
    
    def predict_outputs(self, outputs, temperature):
        outputs = self.softmax(outputs.div(temperature))
        outputs = torch.multinomial(outputs, 1)
        return outputs

    def print_losses(self, epoch, time_elapsed):
        print '[epoch {}] {}'.format(epoch, time_elapsed)
        print 'training loss: {}'.format(self.train_losses[-1])
        print 'validation loss: {}'.format(self.valid_losses[-1])
    
    def validate(self):
        seqs = next(self.loaders['valid'])
        loss = self.evaluate_loss(seqs, self.loaders['valid'].batch_size)
        self.valid_losses.append(loss.data[0])

    def simulate(self, primer=[ord('`')], temperature=1.0):
        """
        To generate music.
        """
        START_SYMBOL = ord('`')
        END_SYMBOL = ord('$')
        softmax = nn.Softmax()
        assert primer[0] == START_SYMBOL
        generated = primer[:]
        
        seqs = [primer]
        inputs, _ = self.prepare_inputs_targets(seqs, requires_grad=False)
        self.net.hidden = self.prepare_hidden(1, requires_grad=False)
        outputs = self.net(inputs, [len(primer)], per_char_generation=True)  # teaching force in guided generation
        self.net.update_hidden()
        output = self.predict_outputs(outputs, temperature).view(-1).cpu().data[-1]
        output = self.corpus.get(output)
        generated.append(output)
        while output != END_SYMBOL:
            seqs = [[output]]
            inputs, _ = self.prepare_inputs_targets(seqs, requires_grad=True)
            outputs = self.net(inputs, [1], per_char_generation=True)
            self.net.update_hidden()
            output = self.predict_outputs(outputs, temperature).view(-1).cpu().data[0]
            output = self.corpus.get(output)
            generated.append(output)
        return generated
        
class CurriculumTrainer(Trainer):
    def __init__(self, net, loaders, corpus, criterion, optimizer, curriculum,
                 cuda, log_period=True, gen_period=True):
        """
        :param curriculum: function that accepts `epoch` and returns `len_lims`
        """
        Trainer.__init__(self, net, loaders, corpus, criterion, optimizer, cuda,
                         log_period=log_period, gen_period=gen_period)
        self.curriculum = curriculum

    def evaluate_loss(self, seqs, batch_size):
        seq_lens = map(len, seqs)
        inputs, targets = self.prepare_inputs_targets(seqs, requires_grad=False)
        self.net.hidden = self.prepare_hidden(batch_size, requires_grad=False)
        outputs = self.net(inputs, seq_lens)
        assert outputs.size(0) == sum(seq_lens) - len(seq_lens), \
            'outputs.size(0) = {} != len(seq_lens) = {}'.format(
                outputs.size(0), sum(seq_lens) - len(seq_lens))
        # TODO: why mustn't I normalize over N?
        loss = self.criterion(outputs, targets)# / (sum(seq_lens)-len(seq_lens))
        return loss
    
    def train(self, max_epoch):
        start = time.time()
        for ep in range(max_epoch):
            self.loaders['train'].len_lims = self.curriculum(ep)
            seqs = next(self.loaders['train'])
            loss = self.evaluate_loss(seqs, self.loaders['train'].batch_size)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.train_losses.append(loss.data[0])

            self.validate()

            if self.log_period is not None and not ep % self.log_period:
                self.print_losses(ep, time.time() - start)
            if self.gen_period is not None and not ep % self.gen_period:
                music = ''.join(map(chr, self.simulate())[1:-1])
                if ep not in self.samples:
                    self.samples[ep] = list()
                self.samples[ep].append(music)
            
        self.epoch += ep
