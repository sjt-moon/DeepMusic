#!/opt/conda/bin/python
import pickle
import numpy as np
import argparse
import sys
from utils import TwoWayDictionary


def make_parser():
    parser = argparse.ArgumentParser(description='Inputs.txt preprocessor.')
    parser.add_argument('-c', '--corpus', action='store_true', help='export the corpus, a list containing '
                        'the int counterpart of each unique character in ABC dataset, as "corpus.p"')
    parser.add_argument('--dict-corpus', action='store_true', help='use two-way dictionary instead of list'
                        ' to store the corpus; dictionary usage: index = corpus[vocabulary]; '
                        'vocabulary = corpus.get(index)', dest='corpus_as_dict')
    parser.add_argument('-z', '--zscore', action='store_true', help='enable zscore; default to disabled')
    parser.add_argument('--mean-std', metavar='FILE', help='output mean and standard deviation to pickle FILE'
                        ' in format {"mean":xxx, "std":xxx}; effective only if `--zscore`', dest='mean_std_file')
    parser.add_argument('--start-symbol', dest='tune_start_symbol', default='`', help='a special character '
                        'marking the beginning of a tune; default to \'%(default)s\'', metavar='CHAR')
    parser.add_argument('--end-symbol', dest='tune_end_symbol', default='$', help='a special character '
                        'marking the ending of a tune; default to \'%(default)s\'', metavar='CHAR')
    parser.add_argument('--dos2unix', action='store_true', help='replace "\\r\\n" with "\\n"; default to disabled')
    parser.add_argument('outputfile', help='the output pickle file; default to "inputs.p"')
    return parser


args = make_parser().parse_args(sys.argv[1:])
tune_start_symbol = args.tune_start_symbol
tune_end_symbol = args.tune_end_symbol
zscore = args.zscore
dos2unix = args.dos2unix
mean_std_tofile = args.mean_std_file
outputfile = args.outputfile
corpusoutfile = 'corpus.p' if args.corpus else None


inputs = []
with open('input.txt') as infile:
    recording = False
    for line in infile:
        line = line.rstrip()
        if line == '<start>':
            cbuf = []
            recording = True
        elif line == '<end>':
            inputs.append(tune_start_symbol + '\n'.join(cbuf) + tune_end_symbol)
            recording = False
        else:
            cbuf.append(line)

if dos2unix:
    for i in range(len(inputs)):
        inputs[i] = inputs[i].replace('\\r\\n', '\\n')

# convert from characters to integers
for i in range(len(inputs)):
    inputs[i] = map(ord, inputs[i])

if corpusoutfile:
    agg_inputs = reduce(lambda a, b: a + b, inputs, [])
    agg_inputs = list(set(agg_inputs))
    agg_inputs.sort()
    
    if args.corpus_as_dict:
        corp = TwoWayDictionary()
        for idx, v in enumerate(agg_inputs):
            corp[v] = idx
    else:
        corp = agg_inputs
    
    with open(corpusoutfile, 'wb') as outfile:
        pickle.dump(corp, outfile)

if zscore:
    # z-scoring
    agg_inputs = np.array(reduce(lambda a, b: a + b, inputs, []))
    m = np.mean(agg_inputs)
    s = np.std(agg_inputs)
    for i in range(len(inputs)):
        for j in range(len(inputs[i])):
            inputs[i][j] = (inputs[i][j] - m) * 1.0 / s

    if mean_std_tofile:
        with open(mean_std_tofile, 'wb') as outfile:
            pickle.dump(dict(mean=m, std=s), outfile)

# save as pickle file
with open(outputfile, 'wb') as outfile:
    pickle.dump(inputs, outfile)
