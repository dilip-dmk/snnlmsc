from __future__ import print_function

import argparse
import datetime
import os
import time

import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle

from rnnlm import RNNLM
from utils import SequenceLoader


def train(args):
    print(vars(args))

    loader = SequenceLoader(args)

    if args.init_from is not None:
        if os.path.isdir(args.init_from):
            assert os.path.exists(args.init_from), "{} is not a directory".format(args.init_from)
            parent_dir = args.init_from
        else:
            assert os.path.exists("{}.index".format(args.init_from)), "{} is not a checkpoint".format(args.init_from)
            parent_dir = os.path.dirname(args.init_from)

        config_file = os.path.join(parent_dir, "config.pkl")
        vocab_file = os.path.join(parent_dir, "vocab.pkl")

        assert os.path.isfile(config_file), "config.pkl does not exist in directory {}".format(parent_dir)
        assert os.path.isfile(vocab_file), "vocab.pkl does not exist in directory {}".format(parent_dir)

        if os.path.isdir(args.init_from):
            checkpoint = tf.train.latest_checkpoint(parent_dir)
            assert checkpoint, "no checkpoint in directory {}".format(args.init_from)
        else:
            checkpoint = args.init_from

        with open(os.path.join(parent_dir, 'config.pkl'), 'rb') as f:
            saved_args = pickle.load(f)

        assert saved_args.hidden_size == args.hidden_size, "hidden size argument ({}) differs from save ({})" \
            .format(saved_args.hidden_size, args.hidden_size)
        assert saved_args.num_layers == args.num_layers, "number of layers argument ({}) differs from save ({})" \
            .format(saved_args.num_layers, args.num_layers)

        with open(os.path.join(parent_dir, 'vocab.pkl'), 'rb') as f:
            saved_vocab = pickle.load(f)

        assert saved_vocab == loader.vocab, "vocab in data directory differs from save"

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    new_config_file = os.path.join(args.save_dir, 'config.pkl')
    new_vocab_file = os.path.join(args.save_dir, 'vocab.pkl')

    if not os.path.exists(new_config_file):
        with open(new_config_file, 'wb') as f:
            pickle.dump(args, f)
    if not os.path.exists(new_vocab_file):
        with open(new_vocab_file, 'wb') as f:
            pickle.dump(loader.vocab, f)

    model = RNNLM(args)



    with tf.Session() as sess:
        #tf.summary.merge_all()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        if args.init_from is not None:
            try:
                saver.restore(sess, checkpoint)
            except ValueError:
                print("{} is not a valid checkpoint".format(checkpoint))

                print("initializing from {}".format(checkpoint))

        start_datetime = datetime.datetime.now().isoformat()

        # if args.tensorboard:
        train_writer = tf.summary.FileWriter(os.path.join(args.save_dir, start_datetime))
        train_writer.add_graph(sess.graph)

        for e in range(args.num_epochs):
            if e % args.decay_every == 0:
                lr = args.learning_rate * (args.decay_factor ** e)

            state = sess.run(model.zero_state)

            for b, (x, y) in enumerate(loader.train):
                global_step = e * loader.train.num_batches + b
                start = time.time()
                feed = {model.x: x,
                        model.y: y,
                        model.dropout: args.dropout,
                        model.lr: lr}
                state_feed = {pl: s for pl, s in zip(sum(model.start_state, ()), sum(state, ()))}
                feed.update(state_feed)
                train_loss, state, _ = sess.run([model.loss, model.end_state, model.train_op], feed)
                end = time.time()

                # if args.verbose:
                print("{}/{} (epoch {}), train_loss = {:.3f}, perplexity = {:.3f}, time/batch = {:.3f}".format(
                    global_step, args.num_epochs * loader.train.num_batches, e, train_loss, np.exp(train_loss),
                    end - start))

                # if args.tensorboard:
                summary = tf.Summary(
                    value=[tf.Summary.Value(tag="RNNLM Train Loss", simple_value=float(train_loss))])

                train_writer.add_summary(summary, global_step)

                summary1 = tf.Summary(
                    value=[tf.Summary.Value(tag="RNNLM Train Perplexity", simple_value=float(np.exp(train_loss)))])
                train_writer.add_summary(summary1, global_step)

                if global_step % args.save_every == 0 \
                        or (e == args.num_epochs - 1 and b == loader.train.num_batches - 1):
                    all_loss = 0
                    val_state = sess.run(model.zero_state)
                    start = time.time()

                    for b, (x, y) in enumerate(loader.val):
                        feed = {model.x: x,
                                model.y: y}
                        state_feed = {pl: s for pl, s in zip(sum(model.start_state, ()), sum(val_state, ()))}
                        feed.update(state_feed)
                        batch_loss, val_state = sess.run([model.loss, model.end_state], feed)
                        all_loss += batch_loss

                    end = time.time()
                    val_loss = all_loss / loader.val.num_batches

                    # if args.verbose:
                    print("val_loss = {:.3f}, perplexity = {:.3f}, time/val = {:.3f}".format(val_loss,
                                                                                             np.exp(val_loss),
                                                                                             end - start))

                    checkpoint_path = os.path.join(args.save_dir, '{}-iter_{}-val_{:.3f}.ckpt' \
                                                   .format(start_datetime, global_step, val_loss))
                    saver.save(sess, checkpoint_path)

                    # if args.verbose:
                    print("model saved to {}".format(checkpoint_path))

                    # if args.tensorboard:
                    summary = tf.Summary(
                        value=[tf.Summary.Value(tag="RNNLM Val Loss", simple_value=float(val_loss))])
                    tf.summary.histogram('Val_Perplexity', val_loss)
                    # tf.histogram_summary('Val_Perplexity', val_loss)
                    train_writer.add_summary(summary, global_step)

                    summary1 = tf.Summary(
                        value=[tf.Summary.Value(tag="RNNLM Val Perplexity", simple_value=float(np.exp(val_loss)))])
                    # tf.summary.histogram('Val_Perplexity', np.exp(val_loss))
                    train_writer.add_summary(summary1, global_step)
        tf.summary.merge_all()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='preprocessed1', help="directory with processed data")
    parser.add_argument('--init_from', type=str,
                        default=None,
                        help="checkpoint file or directory to intialize from, if directory the most recent checkpoint is used")
    parser.add_argument('--save_every', type=int,
                        default=1024, help="batches per save, also reports loss on validation set")
    parser.add_argument('--save_dir', type=str,
                        default='checkpoints_wlm', help="directory to save checkpoints and config files")
    parser.add_argument('--num_epochs', type=int,
                        default=5, help="number of epochs to train")
    parser.add_argument('--batch_size', type=int,
                        default=1000, help="minibatch size")
    parser.add_argument('--vocab_size', type=int,
                        default=None, help="vocabulary size, defaults to infer from the input")
    parser.add_argument('--seq_length', type=int,
                        default=50, help="sequence length")
    parser.add_argument('--level', type=str, default='word',
                        help='word or char')
    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm, or nas')
    parser.add_argument('--learning_rate', type=float,
                        default=2e-3, help="learning rate")
    parser.add_argument('--decay_factor', type=float,
                        default=0.97, help="learning rate decay factor")
    parser.add_argument('--decay_every', type=int,
                        default=10, help="how many epochs between every application of decay factor")
    parser.add_argument('--grad_clip', type=float,
                        default=5, help="maximum value for gradients, set to 0 to remove gradient clipping")
    parser.add_argument('--hidden_size', type=int,
                        default=128, help="size of hidden units in network")
    parser.add_argument('--num_layers', type=int,
                        default=3, help="number of hidden layers in network")
    parser.add_argument('--dropout', type=float,
                        default=0.9, help="dropout keep probability applied to input between lstm layers")
    parser.add_argument('--verbose', action='store_true',
                        help="verbose printing")
    parser.add_argument('--tensorboard', action='store_true',
                        help="tensorboard logging to checkpoint directory")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    train(args)
