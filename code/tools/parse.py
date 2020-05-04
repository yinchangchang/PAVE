# coding=utf8

import argparse

parser = argparse.ArgumentParser(description='DII Challenge 2019')

parser.add_argument(
        '--data-dir',
        type=str,
        default='../data/',
        help='data directory'
        )
parser.add_argument(
        '--result-dir',
        type=str,
        default='../result/',
        help='result directory'
        )
parser.add_argument(
        '--file-dir',
        type=str,
        default='../file/',
        help='useful file directory'
        )
parser.add_argument(
        '--vital-file',
        type=str,
        default='../data/data.csv',
        help='vital information'
        )
parser.add_argument(
        '--master-file',
        type=str,
        default='../data/demo.csv',
        help='master information'
        )
parser.add_argument(
        '--label-file',
        type=str,
        default='../data/label.csv',
        help='label'
        )
parser.add_argument(
        '--model',
        '-m',
        type=str,
        default='attention',
        # default='lstm',
        help='model'
        )
parser.add_argument(
        '--embed-size',
        metavar='EMBED SIZE',
        type=int,
        default=512,
        help='embed size'
        )
parser.add_argument(
        '--rnn-size',
        metavar='rnn SIZE',
        type=int,
        help='rnn size'
        )
parser.add_argument(
        '--hidden-size',
        metavar='hidden SIZE',
        type=int,
        help='hidden size'
        )
parser.add_argument(
        '--n-split',
        metavar='split num',
        type=int,
        default=20,
        help='split num'
        )
parser.add_argument(
        '--split-num',
        metavar='split num',
        type=int,
        default=5,
        help='split num'
        )
parser.add_argument(
        '--split-nor',
        metavar='split normal range',
        type=int,
        default=3,
        help='split num'
        )
parser.add_argument(
        '--num-layers',
        metavar='num layers',
        type=int,
        default=1,
        help='num layers'
        )
parser.add_argument(
        '--num-code',
        metavar='num codes',
        type=int,
        default=1200,
        help='num code'
        )
parser.add_argument(
        '--use-gp',
        type=int,
        default=1
        )
parser.add_argument(
        '--use-glp',
        metavar='use global pooling operation',
        type=int,
        default=0,
        help='use global pooling operation'
        )
parser.add_argument(
        '--use-visit',
        metavar='use visit as input',
        type=int,
        default=1,
        help='use visit or code as input'
        )
parser.add_argument(
        '--use-value',
        metavar='use value embedding as input',
        type=int,
        default=1,
        help='use value embedding as input'
        )
parser.add_argument(
        '--use-cat',
        metavar='use cat for time and value embedding',
        type=int,
        default=1,
        help='use cat or add'
        )
parser.add_argument(
        '--use-trend',
        metavar='use feature variation trend',
        type=int,
        default=0,
        help='use trend'
        )
parser.add_argument(
        '--avg-time',
        metavar='avg time for trend, hours',
        type=int,
        default=4,
        help='avg time for trend'
        )
parser.add_argument(
        '--seed',
        metavar='seed',
        type=int,
        default=1,
        help='seed'
        )
parser.add_argument(
        '--set',
        metavar='split set for training',
        type=int,
        default=0,
        help='split set'
        )
parser.add_argument(
        '--last-time',
        metavar='last-time ',
        type=int,
        default=-10,
        help='last time'
        )
parser.add_argument(
        '--final',
        metavar='final test to submit',
        type=int,
        default=0,
        help='final'
        )




parser.add_argument('--phase',
        default='train',
        type=str,
        metavar='S',
        help='pretrain/train/test phase')
parser.add_argument(
        '--batch-size',
        '-b',
        metavar='BATCH SIZE',
        type=int,
        default=16,
        help='batch size'
        )
parser.add_argument('--save-dir',
        default='../../data',
        type=str,
        metavar='S',
        help='save dir')
parser.add_argument('--resume',
        default='',
        type=str,
        metavar='S',
        help='start from checkpoints')
parser.add_argument('--task',
        default='mortality',
        # default='phy',
        # default='sepsis',
        type=str,
        metavar='S',
        help='start from checkpoints')
parser.add_argument(
        '--time-range',
        default=2160,
        type=int)
parser.add_argument(
        '--compute-weight',
        default=1,
        type=int,
		help='compute weight for interpretebility')
parser.add_argument(
        '--n-code',
        default=8,
        type=int,
		help='at most n codes for same visit'
		)
parser.add_argument(
        '--n-visit',
        default=60,
        type=int,
		help='at most input n visits'
		)

#####
parser.add_argument('-j',
        '--workers',
        default=8,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 32)')
parser.add_argument('--lr',
        '--learning-rate',
        default=0.001,
        type=float,
        metavar='LR',
        help='initial learning rate')
parser.add_argument('--epochs',
        default=20,
        type=int,
        metavar='N',
        help='number of total epochs to run')
parser.add_argument('--save-freq',
        default='5',
        type=int,
        metavar='S',
        help='save frequency')
parser.add_argument('--save-pred-freq',
        default='10',
        type=int,
        metavar='S',
        help='save pred clean frequency')
parser.add_argument('--val-freq',
        default='5',
        type=int,
        metavar='S',
        help='val frequency')
args = parser.parse_args()
