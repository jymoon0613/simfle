import argparse
import os
import gc

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model.model import SimFLE
from util.util import save_checkpoint
from engine_linear import Classifier, train, validate

parser = argparse.ArgumentParser(description='SimFLE Training')

parser.add_argument('--train-data-path', default=None, type=str, dest='train_data_path',
                    help='path to train dataset (default: None)')
parser.add_argument('--test-data-path', default=None, type=str, dest='test_data_path',
                    help='path to test dataset (default: None)')
parser.add_argument('--n-workers', default=10, type=int, dest='n_workers',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--resume', default=None, type=str, dest='resume',
                    help='path to checkpoint for resumption (default: None)')
parser.add_argument('--start-epoch', default=0, type=int, dest='start_epoch',
                    help='start epoch to run (default: 0)')
parser.add_argument('--total-epoch', default=100, type=int, dest='total_epoch',
                    help='number of epochs to run (default: 100)')
parser.add_argument('--batch-size', default=512, type=int, dest='batch_size',
                    help='batch-size for training (default: 512)')
parser.add_argument('--lr', default=0.5, type=float, dest='lr',
                     help='initial learning rate (default: 0.5)')
parser.add_argument('--momentum', default=0.9, type=float, dest='momentum',
                    help='momentum of SGD solver (default: 0.9)')
parser.add_argument('--weight-decay', default=0., type=float, dest='weight_decay',
                    help='weight decay (default: 0.)')
parser.add_argument('--n-gpus', default=1, type=int, dest='n_gpus',
                    help='number of gpus to use (default: 1)')
parser.add_argument('--gpu', default=None, type=int, dest='gpu',
                    help='GPU id to use (default: None)')
parser.add_argument('--print-freq', default=10, type=int, dest='print_freq',
                    help='print frequency (default: 10)')
parser.add_argument('--evaluate', default=False, dest='evaluate',
                    help='evaluate model on validation set (default: False)')

parser.add_argument('--pretrained', type=str, dest='pretrained',
                    help='path to simfle pretrained checkpoint')
parser.add_argument('--n-classes', default=8, type=int, dest='n_classes',
                    help='number of classes to classify (default: 8)')

def main():
    args = parser.parse_args()

    print("Creating model...")

    cudnn.benchmark = True
    gc.collect()
    torch.cuda.empty_cache()

    args.n_groups = 0
    args.mask_ratio = 0

    backbone = SimFLE(args)

    if args.pretrained is not None:
        if os.path.isfile(args.pretrained):
            print("Loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            backbone.load_state_dict(state_dict)
            
            backbone = backbone.online_network.encoder

            model = Classifier(backbone, args)

            print("Loaded SimFle pretrained model '{}'".format(args.pretrained))
        else:
            print("No checkpoint found at '{}'".format(args.pretrained))

    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    if args.n_gpus != 0:
        if args.gpu == None:
            model = torch.nn.DataParallel(model, device_ids=list(range(args.n_gpus))).cuda()
        else:
            model = torch.nn.DataParallel(model, device_ids=[args.n_gpus]).cuda()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

    normalize = transforms.Normalize(mean = [0.5795, 0.4522, 0.3957], std = [0.2769, 0.2473, 0.2412])

    train_dataset = datasets.ImageFolder(
        args.train_data_path,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.n_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.test_data_path, transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=args.n_workers, pin_memory=True)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    init_lr = args.lr * args.batch_size / 256

    optimizer = torch.optim.SGD(parameters, lr=init_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss().cuda()

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
        
    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    print("Training the model...")

    best_acc = 0

    for epoch in range(args.start_epoch, args.total_epoch):

        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        with torch.no_grad():
            acc = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)

        if model.__class__.__name__ == 'DataParallel':
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)

if __name__ == '__main__':
    main()