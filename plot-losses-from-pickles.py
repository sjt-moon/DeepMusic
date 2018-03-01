#!/usr/bin/env python
import pickle
import models
from utils import TwoWayDictionary
import sys
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def make_parser():
    parser = argparse.ArgumentParser(description='Create loss plots from dumped pickle file')
    parser.add_argument('filenames', metavar='filename', nargs='*')
    return parser

def main():
    args = make_parser().parse_args(sys.argv[1:])
    for filename in args.filenames:
        with open(filename, 'rb') as infile:
            _, train_losses, valid_losses, _ = pickle.load(infile)
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses)
        plt.plot(valid_losses)
        plt.legend(('train', 'validation'))
        plt.xlabel('# of batches')
        plt.ylabel('losses')
        plt.grid(True)
        plt.savefig(os.path.splitext(filename)[0] + '.png')

if __name__ == '__main__':
    main()
