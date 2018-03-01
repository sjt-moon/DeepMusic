import trainers
import models
import pickle
import trainlib
import dataloaders as dl
import torch
import torch.optim as optim
import torch.nn as nn
from utils import TwoWayDictionary
with open('inputs.p', 'rb') as infile:
    inputs = pickle.load(infile)
with open('corpus.p', 'rb') as infile:
    corp = pickle.load(infile)
trainset, validset = trainlib.divide_train_validation(inputs)

def curriculum(epoch):
    if epoch < 400:
        seqs_len = (25, 30)
    elif epoch < 800:
        seqs_len = (60, 120)
    else:
        seqs_len = (150, 200)
    return seqs_len

net = models.MusicGenNet(75, 95,
                 n_layers=3,
                 using_gru=False,
                 dropout=0.1)
net = net.cuda()
loaders = {
    'train': dl.AcrossTuneNonOverlapDataloader(trainset, (25, 30), batch_size=16, strict_len_lims=False),
    'valid': dl.ValidationDataloader(validset)
}
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters(), lr=0.004)
curr_trainer = trainers.CurriculumTrainer(net, loaders, corp, criterion, optimizer, curriculum,
                 True, log_period=True, gen_period=True)

curr_trainer.train(1000)

for ep in curr_trainer.samples:
    for i, music in enumerate(curr_trainer.samples[ep]):
        with open('out/ep{}-{}.abc'.format(ep, i), 'w') as outfile:
            outfile.write(music)

for i in range(10):
    with open('out/final-{}.abc'.format(i), 'w') as outfile:
        music = ''.join(map(chr, curr_trainer.simulate(primer=map(ord, '`'), temperature=1))[1:-1])
        outfile.write(music)

with open('out/fc_bn_100unit_2layer_lstm_0_2dropout_net+losses+music.p', 'wb') as outfile:
    pickle.dump((net.cpu(), curr_trainer.train_losses, curr_trainer.valid_losses, curr_trainer.samples), outfile)

