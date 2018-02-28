import time
import trainlib
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
import itertools


def prepare_inputs(corpus, seqs, cuda=False, requires_grad=False):
    inputs = trainlib.one_hot_tensor(corpus, seqs)
    if cuda:
        inputs = Variable(inputs.cuda(), requires_grad=requires_grad)
    else:
        inputs = Variable(inputs, requires_grad=requires_grad)
    return inputs


class Simulator:
    def __init__(self, model, corpus):
        self.model = model
        self.corpus = corpus

    def run(self, primer):
        """
        :param primer: a string primer
        :return: the generated string
        """
        pass


class Trainer:
    def __init__(self, net, loaders, corpus, criterion, optimizer,
                 cuda, log_period):
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

        self.epoch = 0  # accumulated epoch
        self.train_losses = list()
        self.valid_losses = list()
        self.samples = dict()  # { epoch: samples }

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
        if self.cuda:
            hidden = (hidden[0].cuda(), hidden[1].cuda())
        hidden = (Variable(hidden[0], requires_grad=requires_grad),
                  Variable(hidden[1], requires_grad=requires_grad))
        return hidden

    def print_losses(self, epoch, time_elapsed):
        print '[epoch {}] {}'.format(epoch, time_elapsed)
        print 'training loss: {}'.format(self.train_losses[-1])
        print 'validation loss: {}'.format(self.valid_losses[-1])

    def simulate(self):
        # TODO
        pass


class CurriculumTrainer(Trainer):
    def __init__(self, net, loaders, corpus, criterion, optimizer, curriculum,
                 cuda, log_period):
        """
        :param curriculum: function that accepts `epoch` and returns `len_lims`
        """
        Trainer.__init__(self, net, loaders, corpus, criterion, optimizer, cuda,
                         log_period)
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

            if not ep % self.log_period:
                self.print_losses(ep, time.time() - start)
        self.epoch += ep

    def validate(self):
        seqs = next(self.loaders['valid'])
        loss = self.evaluate_loss(seqs, self.loaders['valid'].batch_size)
        self.valid_losses.append(loss.data[0])
