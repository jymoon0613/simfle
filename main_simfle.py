import argparse
import os
import time
import gc
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from model.model import SimFLE
from model.loss import DistillationLoss, SimilarityLoss
from data.dataset import SimFLEDataset
from util.util import adjust_learning_rate, get_n_params

parser = argparse.ArgumentParser(description='SimFLE Training')

parser.add_argument('--data-path', default=None, type=str, dest='data_path',
                    help='path to dataset')
parser.add_argument('--n-workers', default=10, type=int, dest='n_workers',
                    help='number of data loading workers')
parser.add_argument('--epochs', default=100, type=int, dest='epochs',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, dest='batch_size',
                    help='batch-size for training')
parser.add_argument('--lr', default=0.05, type=float, dest='lr',
                     help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, dest='momentum',
                    help='momentum of SGD solver')
parser.add_argument('--weight-decay', default=1e-4, type=float, dest='weight_decay',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--n-gpus', default=0, type=int, dest='n_gpus',
                    help='number of gpus to use')
parser.add_argument('--gpu', default=None, type=int, dest='gpu',
                    help='GPU id to use')

parser.add_argument('--mask-ratio', default=0.75, type=float, dest='mask_ratio',
                    help='mask ratio for semantic masking')
parser.add_argument('--n-groups', default=32, type=int, dest='n_groups',
                    help='number of groups for channel grouping')

def main():
    args = parser.parse_args()
    print(args)

    print("Creating model")

    model = SimFLE(args)

    if args.n_gpus != 0:
        if args.gpu == None:
            torch.nn.DataParallel(model, device_ids=list(range(args.n_gpus)))
        else:
            torch.nn.DataParallel(model, device_ids=[args.gpu])
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    train_dataset = SimFLEDataset(args.data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=args.n_workers, pin_memory=True, drop_last=False, shuffle=True)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    init_lr = args.lr * args.batch_size / 256

    optimizer = torch.optim.SGD(parameters, lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

if __name__ == '__main__':
    main()
