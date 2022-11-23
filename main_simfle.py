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
                    help='number of total epochs to run (default: 100)')
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

def main():
    args = parser.parse_args()

    print("Creating model...")

    cudnn.benchmark = True
    gc.collect()
    torch.cuda.empty_cache()

    model = SimFLE(args)

    if args.n_gpus != 0:
        if args.gpu == None:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpus))).cuda()
        else:
            model = torch.nn.DataParallel(model, device_ids=[args.n_gpus]).cuda()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    train_dataset = SimFLEDataset(args.data_path)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                  num_workers=args.n_workers, pin_memory=True, drop_last=False, shuffle=True)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    init_lr = args.lr * args.batch_size / 256

    optimizer = torch.optim.SGD(parameters, lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                dev = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=dev)

            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            if model.__class__.__name__ == 'DataParallel':
                model.module.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint['state_dict'])
            print("Loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    print("Training the model...")

    for epoch in range(args.start_epoch, args.total_epoch):

        adjust_learning_rate(optimizer, init_lr, epoch, args.max_epoch)

        train_simfle(train_loader, model, optimizer, epoch, args)
        if model.__class__.__name__ == 'DataParallel':
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer' : optimizer.state_dict()},
                is_best=False, filename='checkpoint_{:03d}.pth.tar'.format(epoch))
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()},
                is_best=False, filename='checkpoint_{:03d}.pth.tar'.format(epoch))

if __name__ == '__main__':
    main()
