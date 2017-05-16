from __future__ import print_function
import argparse
import os
import numpy as np
import codecs
import timeit
import collections
from six.moves import cPickle as pickle

parser = argparse.ArgumentParser()
# parser.add_argument('--input_file', type=str,
#                     default='../Vulnerability-Detection-PyCodes/misc/tokresults.txt', help="data file with tokens")
parser.add_argument('--input_file', type=str,
                    default='data/tinyshakespeare/input.txt', help="data file with tokens")
parser.add_argument('--data_dir', type=str,
                    default='preprocessed1', help="save directory for preprocessed files")
parser.add_argument('--val_frac', type=float,
                    default=0.1, help="fraction of data to use as validation set")
# parser.add_argument('--test_frac', type=float,
#                     default=0.1, help="fraction of data to use as test set")
parser.add_argument('--verbose', action='store_true',
                    help="verbose printing")

start = timeit.default_timer()
args = parser.parse_args()

if not os.path.exists(args.data_dir):
    os.makedirs(args.data_dir)

vocab_file = os.path.join(args.data_dir, "vocab.pkl")
train_file = os.path.join(args.data_dir, "train.npy") #train_tensor_file
val_file = os.path.join(args.data_dir, "val.npy") #valid_tensor_file
#test_file = os.path.join(args.data_dir, "test.npy")

print("Reading {} file...".format(args.input_file))
with codecs.open(args.input_file, 'r', encoding='ISO-8859-1') as f:
    data = f.read()

counter = collections.Counter(data)
counts = sorted(counter.items(), key=lambda x: -x[1])

tokens, _ = zip(*counts)
vocab_size = len(tokens)

vocab = dict(zip(tokens, range(vocab_size)))

print("Writing vocab.pkl file...")
with open(vocab_file, 'wb') as f:
    pickle.dump(vocab, f)

data_tensor = np.array(list(map(vocab.get, data)))
data_size = data_tensor.shape[0]

print("Splitting into Training and Validation set...")
val_size = int(args.val_frac * data_size)
#test_size = int(args.test_frac * data_size)
train_size = data_size - val_size #- test_size

np.save(train_file, data_tensor[:train_size])
np.save(val_file, data_tensor[train_size:train_size + val_size])
#np.save(test_file, data_tensor[train_size + val_size:data_size])

print("++++ Preprocess Done ++++")
print("Vocab size: {}, Data size: {}, Train size: {}, Validation size: {}".format(vocab_size, data_size, train_size, val_size))
#print("vocab and frequency rank: {}".format(vocab))
stop = timeit.default_timer()
print("** Elapsed time : %8.2f seconds **" % (stop - start))
