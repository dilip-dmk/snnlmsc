from __future__ import print_function
import os
import numpy as np
from six.moves import cPickle as pickle


class SequenceLoader():

    def __init__(self, args):
        self.args = args

        vocab_file = os.path.join(self.args.data_dir, "vocab.pkl")
        train_file = os.path.join(args.data_dir, "train.npy")
        val_file = os.path.join(args.data_dir, "val.npy")
        #test_file = os.path.join(args.data_dir, "test.npy")

        assert os.path.exists(args.data_dir), "data directory does not exist"
        assert os.path.exists(vocab_file), "vocab file does not exist"
        assert os.path.exists(train_file), "train file does not exist"
        assert os.path.exists(val_file), "validation file does not exist"
        #assert os.path.exists(test_file), "test file does not exist"

        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)

        if args.vocab_size is None: # infer vocab size from data
            args.vocab_size = len(self.vocab)
        else:
            assert args.vocab_size == len(self.vocab), \
                "specified vocab size ({}) is not the same as data ({}), remove flag to infer the size" \
                    .format(args.vocab_size, len(self.vocab))

        sets = {}

        for label, file in [("train", train_file), ("validation", val_file)]:
            data_tensor = np.load(file)
            num_batches = int(data_tensor.size / (args.batch_size * args.seq_length))

            assert num_batches != 0, \
                "not enough {} data for batch parameters. reduce seq_length or batch_size or preprocess with different splits" \
                .format(label)

            data_tensor = data_tensor[:num_batches * args.batch_size * args.seq_length]

            x = data_tensor
            y = np.empty(data_tensor.shape, dtype=data_tensor.dtype)
            y[:-1] = x[1:]
            y[-1] = x[0]

            x_batches = np.split(x.reshape(args.batch_size, -1), num_batches, 1)
            y_batches = np.split(y.reshape(args.batch_size, -1), num_batches, 1)

            sets[label] = BatchIterator(x_batches, y_batches)

            if args.verbose:
                print("{} data loaded".format(label))
                print("Number of batches: {}".format(num_batches))

        self.train = sets["train"]
        self.val = sets["validation"]
       # self.test = sets["test"]


class BatchIterator():

    def __init__(self, x_batches, y_batches):
        self.x_batches = x_batches
        self.y_batches = y_batches
        self.num_batches = len(x_batches)

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter == self.num_batches:
            raise StopIteration
        else:
            x, y = self.x_batches[self.counter], self.y_batches[self.counter] 
            self.counter += 1
            return x, y

    next = __next__ # python 2
