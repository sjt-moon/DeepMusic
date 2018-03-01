#!/usr/bin/env python
import os
import pickle
import matplotlib.pyplot as plt


datadir = 'out'


for filename in filter(lambda x: x.endswith('.p'), os.listdir(datadir)):
    with open(filename, 'rb') as infile:
        _, train_losses, valid_losses, _ = pickle.load(infile)
    plt.figure(figsize=(8, 6))
    plt.xlabel('mini-batches (batch size 128)')
    plt.ylabel('loss')
    plt.plot(train_losses)
    plt.plot(valid_losses)
    plt.grid(True)
    plt.legend(('train', 'validation'))
    plt.savefig(os.path.splitext(filename)[0] + '-losses.png')
