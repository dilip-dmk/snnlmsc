from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import argparse
import os
import random
from six.moves import cPickle as pickle
from six import text_type

from rnnlm import RNNLM


def sample(model, sess, vocab, length, temperature=1.0, prime=None):
    state = sess.run(model.zero_state)

    idx_to_word = {v: k for k, v in vocab.items()}

    if prime:
        for char in prime[:-1]:
            x = np.empty((1, 1))
            x[0, 0] = vocab[char]
            feed = {model.x: x}
            state_feed = {pl: s for pl, s in zip(sum(model.start_state, ()), sum(state, ()))}
            feed.update(state_feed)
            state = sess.run([model.end_state], feed)[0]

    if prime:
        ret = prime
    else:
        ret = str(random.choice(list(vocab.keys())))

    char = ret[-1]

    for _ in range(length):
        x = np.empty((1, 1))
        x[0, 0] = vocab[char]
        feed = {model.x: x}
        state_feed = {pl: s for pl, s in zip(sum(model.start_state, ()), sum(state, ()))}
        feed.update(state_feed)
        logits, state = sess.run([model.logits, model.end_state], feed)
        logits = logits[0]

        if temperature == 0.0:
            sample = np.argmax(logits)
        else:
            scale = logits / temperature
            exp = np.exp(scale - np.max(scale))
            soft = exp / np.sum(exp)

            sample = np.random.choice(len(soft), p=soft)

        pred = idx_to_word[sample]
        ret += pred
        char = pred

    return ret

def setup_and_sample(args):
    if os.path.isdir(args.init_from):
        assert os.path.exists(args.init_from),"{} is not a directory".format(args.init_from)
        parent_dir = args.init_from
    else:
        assert os.path.exists("{}.index".format(args.init_from)),"{} is not a checkpoint".format(args.init_from)
        parent_dir = os.path.dirname(args.init_from)

    config_file = os.path.join(parent_dir, "config.pkl")
    vocab_file = os.path.join(parent_dir, "vocab.pkl")

    assert os.path.isfile(config_file), "config.pkl does not exist in directory {}".format(parent_dir)
    assert os.path.isfile(vocab_file), "vocab.pkl does not exist in directory {}".format(parent_dir)

    with open(config_file, 'rb') as f:
        saved_args = pickle.load(f)

    with open(vocab_file, 'rb') as f:
        saved_vocab = pickle.load(f)

    if os.path.isdir(args.init_from):
        checkpoint = tf.train.latest_checkpoint(parent_dir)
        assert checkpoint, "no checkpoint in directory {}".format(init_from)
    else:
        checkpoint = args.init_from

    saved_args.batch_size = 1
    saved_args.seq_length = 1
    model = RNNLM(saved_args)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        try:
            saver.restore(sess, checkpoint)
        except ValueError:
            print("{} is not a valid checkpoint".format(checkpoint))

        ret = sample(model, sess, saved_vocab, args.length, args.temperature, args.prime)

    return ret

def get_sample_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--init_from', type=str,
                        default='checkpoints', help="checkpoint file or directory to intialize from, if directory the most recent checkpoint is used, directory must have vocab and config files")
    parser.add_argument('--length', type=int, default=500,
                       help="length of desired sample")
    parser.add_argument('--prime', type=text_type, default=u'',
                       help="primer for generation")
    parser.add_argument('--temperature', type=float, default=1,
                       help="sampling temperature")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_sample_args()
    ret = setup_and_sample(args)
    print(ret)
