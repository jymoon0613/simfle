import argparse
import os
import gc

import torch
import torch.backends.cudnn as cudnn

from model.model import SimFLE
from data.dataset import SimFLEDataset
from util.util import adjust_learning_rate, save_checkpoint
from train_engine import train_simfle

parser = argparse.ArgumentParser(description='SimFLE Training')

parser.add_argument('--data-path', default=None, type=str, dest='data_path',
                    help='path to dataset (default: None)')
parser.add_argument('--n-workers', default=10, type=int, dest='n_workers',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--resume', default=None, type=str, dest='resume',
                    help='path to checkpoint for resumption (default: None)')
parser.add_argument('--start-epoch', default=0, type=int, dest='start_epoch',
                    help='start epoch to run (default: 0)')
parser.add_argument('--total-epoch', default=100, type=int, dest='total_epoch',
                    help='number of epochs to run (default: 100)')
parser.add_argument('--max-epoch', default=100, type=int, dest='max_epoch',
                    help='Maximum number of epochs to run (default: 100)')
parser.add_argument('--batch-size', default=256, type=int, dest='batch_size',
                    help='batch-size for training (default: 256)')
parser.add_argument('--lr', default=0.05, type=float, dest='lr',
                     help='initial learning rate (default: 0.05)')
parser.add_argument('--momentum', default=0.9, type=float, dest='momentum',
                    help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight-decay', default=1e-4, type=float, dest='weight_decay',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--n-gpus', default=1, type=int, dest='n_gpus',
                    help='number of gpus to use (default: 1)')
parser.add_argument('--gpu', default=None, type=int, dest='gpu',
                    help='GPU id to use (default: None)')
parser.add_argument('--print-freq', default=10, type=int, dest='print_freq',
                    help='print frequency (default: 10)')

parser.add_argument('--mask-ratio', default=0.75, type=float, dest='mask_ratio',
                    help='mask ratio for semantic masking (default: 0.75)')
parser.add_argument('--n-groups', default=32, type=int, dest='n_groups',
                    help='number of groups for channel grouping (default: 32)')
parser.add_argument('--alpha', default=0.3, type=float, dest='alpha',
                    help='weight for distillation loss (default: 0.3)')
parser.add_argument('--beta', default=0.03, type=float, dest='beta',
                    help='weight for channel grouping loss (default: 0.03)')