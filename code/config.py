
import theano
import argparse


_floatX = theano.config.floatX


def str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1', 'y')


def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', str2bool)

    # Basics
    parser.add_argument('--debug',
                        type='bool',
                        default=False,
                        help='whether it is debug mode')

    parser.add_argument('--test_only',
                        type='bool',
                        default=False,
                        help='test_only: no need to run training process')

    parser.add_argument('--random_seed',
                        type=int,
                        default=1013,
                        help='Random seed')

    # Data file
    parser.add_argument('--train_file',
                        type=str,
                        default=None,
                        help='Training file')

    parser.add_argument('--dev_file',
                        type=str,
                        default=None,
                        help='Development file')

    parser.add_argument('--pre_trained',
                        type=str,
                        default=None,
                        help='Pre-trained model.')

    parser.add_argument('--model_file',
                        type=str,
                        default='model.pkl.gz',
                        help='Model file to save')

    parser.add_argument('--log_file',
                        type=str,
                        default=None,
                        help='Log file')

    parser.add_argument('--embedding_file',
                        type=str,
                        default=None,
                        help='Word embedding file')

    parser.add_argument('--max_dev',
                        type=int,
                        default=None,
                        help='Maximum number of dev examples to evaluate on')

    parser.add_argument('--relabeling',
                        type='bool',
                        default=True,
                        help='Whether to relabel the entities when loading the data')

    # Model details
    parser.add_argument('--embedding_size',
                        type=int,
                        default=None,
                        help='Default embedding size if embedding_file is not given')

    parser.add_argument('--hidden_size',
                        type=int,
                        default=128,
                        help='Hidden size of RNN units')

    parser.add_argument('--bidir',
                        type='bool',
                        default=True,
                        help='bidir: whether to use a bidirectional RNN')

    parser.add_argument('--num_layers',
                        type=int,
                        default=1,
                        help='Number of RNN layers')

    parser.add_argument('--rnn_type',
                        type=str,
                        default='gru',
                        help='RNN type: lstm or gru (default)')

    parser.add_argument('--att_func',
                        type=str,
                        default='bilinear',
                        help='Attention function: bilinear (default) or mlp or avg or last or dot')

    # Optimization details
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size')

    parser.add_argument('--num_epoches',
                        type=int,
                        default=100,
                        help='Number of epoches')

    parser.add_argument('--eval_iter',
                        type=int,
                        default=100,
                        help='Evaluation on dev set after K updates')

    parser.add_argument('--dropout_rate',
                        type=float,
                        default=0.2,
                        help='Dropout rate')

    parser.add_argument('--optimizer',
                        type=str,
                        default='sgd',
                        help='Optimizer: sgd (default) or adam or rmsprop')

    parser.add_argument('--learning_rate', '-lr',
                        type=float,
                        default=0.1,
                        help='Learning rate for SGD')

    parser.add_argument('--grad_clipping',
                        type=float,
                        default=10.0,
                        help='Gradient clipping')

    return parser.parse_args()
