from __future__ import division
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
#from tensorflow.python.ops import rnn_cell
from tensorflow.contrib import rnn as rnn_cell
from tensorflow.contrib import seq2seq
# from tensorflow.contrib import rnn, seq2seq
#from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import variable_scope


class RNNLM():
    def __init__(self, args):
        if args.model == 'rnn':
            layer_type = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            layer_type = rnn_cell.GRUCell
        elif args.model == 'lstm':
            layer_type = rnn_cell.BasicLSTMCell
        elif args.model == 'nas':
            layer_type = tf.contrib.rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        #layer_type = rnn_cell.BasicLSTMCell  #rnn_cell.BasicLSTMCell
        layer = layer_type(args.hidden_size, state_is_tuple=True)
        self.dropout = tf.placeholder_with_default(tf.constant(1, dtype=tf.float32), None)
        wrapped = rnn_cell.DropoutWrapper(layer, input_keep_prob=self.dropout)
        tf.summary.scalar('dropout_input_keep_probability', self.dropout)
        self.core = rnn_cell.MultiRNNCell([wrapped] * args.num_layers, state_is_tuple=True)

        self.x = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.y = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.zero_state = self.core.zero_state(args.batch_size, tf.float32)
        self.start_state = [(tf.placeholder(tf.float32, [args.batch_size, args.hidden_size]), \
                             tf.placeholder(tf.float32, [args.batch_size, args.hidden_size])) \
                            for _ in range(args.num_layers)]

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable('softmax_w', [args.hidden_size, args.vocab_size])
            tf.summary.histogram('softmax_w', softmax_w)
            softmax_b = tf.get_variable('softmax_b', [args.vocab_size])
            tf.summary.histogram('softmax_b', softmax_b)
            embedding = tf.get_variable('embedding', [args.vocab_size, args.hidden_size])

            embedded = tf.nn.embedding_lookup(embedding, self.x)
            if args.level == 'char':
                inputs = tf.unstack(embedded, axis=1)
            elif args.level == 'word':
                inputs = tf.split(axis=1, num_or_size_splits=args.seq_length, value=embedded)
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
            else:
                raise Exception("level type not supported: {}".format(args.model))

            state = self.start_state
            outputs = []
            states = []

            for i, inp in enumerate(inputs):
                if i > 0:
                    variable_scope.get_variable_scope().reuse_variables()

                output, state = self.core(inp, state)
                states.append(state)
                outputs.append(output)

        self.end_state = states[-1]
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, args.hidden_size])

        self.logits = tf.matmul(output, softmax_w) + softmax_b
        hist_out2 = tf.summary.histogram('out2_hist', self.logits)

        self.probs = tf.nn.softmax(self.logits)
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits],
                                                [tf.reshape(self.y, [-1])],
                                                [tf.ones([args.batch_size * args.seq_length])],
                                                args.vocab_size)
        self.loss = tf.reduce_sum(loss) / args.batch_size / args.seq_length

        self.lr = tf.placeholder_with_default(tf.constant(0, dtype=tf.float32), None)
        trainables = tf.trainable_variables()
        grads = tf.gradients(self.loss, trainables)

        if args.grad_clip > 0:
            grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)

        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, trainables))

        tf.summary.histogram('Logits', self.logits)
        tf.summary.histogram('Loss', loss)
        #tf.summary.scalar('Train_Loss', self.cost)
