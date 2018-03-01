#!/usr/bin/env python
import argparse
import pickle
import torch
from torch.autograd import Variable
import models
from utils import TwoWayDictionary
import os
import trainers
import sys


start_symbol = '`'
end_symbol = '$'

def make_parser():
    parser = argparse.ArgumentParser(description='Generate abc files from dumped pickle file. '
                                    'There\'s no start and end symbol in the generated files. '
                                    'The music are named after the pickle file, i.e. '
                                    '`${pickle_filename}-music[-${suffix}]-${index}.abc\'.')
    parser.add_argument('filenames', metavar='filename', nargs='*')
    parser.add_argument('-n', dest='num_music_per_pickle', metavar='N', help='number of abc '
                        'files to generate per pickle file', type=int, default=10)
    parser.add_argument('-p', help='the primer to use; don\'t need to specify the start symbol',
                        dest='primer', default=start_symbol)
    parser.add_argument('-t', dest='temperature', metavar='T', type=float, help='the temperature'
                        ' to generate music', default=1.0)
    parser.add_argument('-s', dest='suffix', help='suffix in abc filenames to differentiate '
                        'between different launches')
    return parser

def preprocess_primer(arg_primer):
    if not arg_primer.startswith(start_symbol):
        arg_primer = ''.join((start_symbol, arg_primer))
    primer = map(ord, arg_primer)
    return primer

def generate(net, corpus, primer, temperature, music_num):
    trainer = trainers.Trainer(net, None, corpus, None, None,
                 False, log_period=False, gen_period=False)
    music_list = list()
    for i in range(music_num):
        music = ''.join(map(chr, trainer.simulate(primer, temperature=temperature)[1:-1]))
        assert type(music) is str, 'wrong type of music'
        music_list.append(music)
    return music_list

def form_music_name(pickle_filename, music_idx, suffix):
    return os.path.splitext(pickle_filename)[0] + '-music' + \
            ('' if not suffix else ('-' + suffix)) + \
            '-' + str(music_idx) + '.abc'

def write_music(pickle_filename, music_list, suffix):
    for i, music in enumerate(music_list):
        with open(form_music_name(pickle_filename, i, suffix), 'w') as outfile:
            outfile.write(music)

def main():
    args = make_parser().parse_args(sys.argv[1:])
    primer = preprocess_primer(args.primer)
    music_num = args.num_music_per_pickle
    temperature = args.temperature
    with open('corpus.p', 'rb') as infile:
        # requires `from utils import TwoWayDictionary` here
        corpus = pickle.load(infile)
    
    for pickle_filename in args.filenames:
        with open(pickle_filename, 'rb') as infile:
            net, _, _, _ = pickle.load(infile)
            music_list = generate(net, corpus, primer, temperature, music_num)
            write_music(pickle_filename, music_list, args.suffix)

if __name__ == '__main__':
    main()

    
    
