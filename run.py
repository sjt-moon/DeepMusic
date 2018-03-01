import itertools
import os
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

def fine_curriculum(epoch):
    if epoch < 200:
        seqs_len = (2, 3)
    elif epoch < 400:
        seqs_len = (4, 5)
    elif epoch < 600:
        seqs_len = (8, 9)
    elif epoch < 800:
        seqs_len = (16, 17)
    else:
        seqs_len = (32, 33)
    return seqs_len

def longfine_curriculum(epoch):
    if epoch < 400:
        seqs_len = (2, 3)
    elif epoch < 800:
        seqs_len = (4, 5)
    elif epoch < 1200:
        seqs_len = (8, 9)
    elif epoch < 1600:
        seqs_len = (16, 17)
    elif epoch < 1800:
        seqs_len = (32, 33)
    else:
        seqs_len = (64, 65)
    return seqs_len

def biasedlongfine_curriculum(epoch):
    if epoch < 800:
        seqs_len = (2, 3)
    elif epoch < 1200:
        seqs_len = (4, 5)
    elif epoch < 1600:
        seqs_len = (16, 17)
    elif epoch < 1800:
        seqs_len = (32, 33)
    else:
        seqs_len = (64, 65)
    return seqs_len

def form_task_name(**kwargs):
    max_epoch = kwargs['max_epoch']
    n_lstm = kwargs['n_lstm']
    n_layers = kwargs['n_layers']
    net_type = 'gru' if kwargs['using_gru'] else 'lstm'
    dropout = str(kwargs['dropout']).replace('.', '_')
    curriculum_name = kwargs['curriculum'].__name__
    curriculum_name = curriculum_name.split('_')[0] if '_' in curriculum_name else 'default'
    return '{}_{}epoch_{}layer_{}unit_{}dropout_{}_net+losses+music'.format(curriculum_name, max_epoch, n_layers,n_lstm,dropout,net_type)


n_lstms = (50, 75, 100, 150,)
n_layerss = (1,)
using_grus = (False,)
dropouts = (0.1, 0.2, 0.3)
max_epochs = (1000, 2200)
cudas = (True,)
batch_sizes = (128,)  # useless field
loaderss = (
    dict(train=dl.AcrossTuneNonOverlapDataloader(trainset, (25, 30), batch_size=128, strict_len_lims=False),
         valid=dl.ValidationDataloader(validset)),
)
optimizers = ((optim.Adam, dict(lr=0.004)),)
trainer_classes = (trainers.CurriculumTrainer,)
curriculums = (curriculum, longfine_curriculum,)

# set filter rule to skip certain job description if the condition returns True
def filterout(job_description):
    skipit = False
    max_epoch = job_description['max_epoch']
    curriculumf = job_description['curriculum']
    
    if curriculumf is curriculum and max_epoch > 1000:
        skipit = True
    elif curriculumf is not curriculum and max_epoch < 1500:
        skipit = True
    elif curriculumf is not curriculum and dropouts
    return skipit



job_descriptions = {(form_task_name(n_lstm=a1,n_layers=a2,using_gru=a3,dropout=a4,max_epoch=a5,cuda=a6,batch_size=a7,loaders=a8,optimizer=a9,trainer_class=a10,curriculum=a11)): dict(n_lstm=a1,n_layers=a2,using_gru=a3,dropout=a4,max_epoch=a5,cuda=a6,batch_size=a7,loaders=a8,optimizer=a9,trainer_class=a10,curriculum=a11) for a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11 in itertools.product(n_lstms,n_layerss,using_grus,dropouts,max_epochs,cudas,batch_sizes,loaderss,optimizers,trainer_classes,curriculums)}




def run(task_name, **kwargs):
    n_lstm = kwargs['n_lstm']
    n_layers = kwargs['n_layers']
    using_gru = kwargs['using_gru']
    dropout = kwargs['dropout']
    max_epoch = kwargs['max_epoch']
    cuda = kwargs['cuda']
    batch_size = kwargs['batch_size']
    loaders = kwargs['loaders']
    trainer_class = kwargs['trainer_class']
    optimizer_class, optimizer_kwargs = kwargs['optimizer']
    curriculum = kwargs['curriculum']
    
    
    net = models.MusicGenNet(n_lstm, 95,
                 n_layers=n_layers,
                 using_gru=using_gru,
                 dropout=dropout)
    if cuda:
        net.cuda()
    
    optimizer = optimizer_class(net.parameters(), **optimizer_kwargs)
    criterion = nn.CrossEntropyLoss()
    trainer = trainer_class(net, loaders, corp, criterion, optimizer, curriculum, cuda, log_period=100, gen_period=True)

    trainer.train(max_epoch)

    if not os.path.isdir('out'):
        os.mkdir('out')

    with open('out/{}.p'.format(task_name), 'wb') as outfile:
        pickle.dump((net.cpu(), trainer.train_losses, trainer.valid_losses, trainer.samples), outfile)


if __name__ == '__main__':
    task_num = len(job_descriptions)
    for i, task_name in enumerate(job_descriptions):
        if filterout(job_descriptions[task_name]):
            print 'skipped', task_name, '({}/{}) =============='.format(i, task_num)
            continue
        print 'begin', task_name, '({}/{}) =============='.format(i, task_num)
        run(task_name, **job_descriptions[task_name])
        print 'end', task_name, '({}/{}) ============='.format(i, task_num)
