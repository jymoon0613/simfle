import argparse
import shutil
import time
import os
import gc

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

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
parser.add_argument('--print-freq', default=10, type=int, dest='print_freq',
                    help='print frequency (default: 10)')

parser.add_argument('--mask-ratio', default=0.75, type=float, dest='mask_ratio',
                    help='mask ratio for semantic masking')
parser.add_argument('--n-groups', default=32, type=int, dest='n_groups',
                    help='number of groups for channel grouping')
parser.add_argument('--alpha', default=0.3, type=float, dest='alpha',
                    help='weight for distillation loss')
parser.add_argument('--beta', default=3e-03, type=float, dest='beta',
                    help='weight for channel grouping loss')

def main():
    args = parser.parse_args()
    print(args)

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

    print("Training the model...")

    for epoch in range(args.epochs):

        adjust_learning_rate(optimizer, init_lr, epoch, args.epochs)

        train(train_loader, model, optimizer, epoch, args)
        if model.__class__.__name__ == 'DataParallel':
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.module.state_dict(),
                'optimizer' : optimizer.state_dict()},
                is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()},
                is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))

def train(train_loader, model, optimizer, epoch, args):

    criterions_s = SimilarityLoss()
    criterions_d = DistillationLoss(T=4)
    criterions_g = torch.nn.MSELoss()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses_s = AverageMeter('SimLoss', ':.4f')
    losses_d = AverageMeter('DistillLoss', ':.4f')
    losses_g = AverageMeter('GroupLoss', ':.4f')
    losses_r = AverageMeter('RecLoss', ':.4f')
    losses_t = AverageMeter('TotalLoss', ':.4f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses_s, losses_d, losses_g, losses_r, losses_t],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (origin, inps) in enumerate(train_loader):

        data_time.update(time.time() - end)

        inps[0] = inps[0].cuda(non_blocking=True)
        inps[1] = inps[1].cuda(non_blocking=True)
        origin = origin.cuda(non_blocking=True)

        g, loss_r, p1, p2, q1, q2, p1_kd, p2_kd, part_kd, _, _ = model(origin, inps)
        g_ = g * 0

        stable_out = torch.cat((p1_kd.unsqueeze(1), p2_kd.unsqueeze(1), part_kd.unsqueeze(1)), dim=1).mean(dim=1)
        stable_out = stable_out.detach()

        loss_s = criterions_s(p1, q1) + criterions_s(p2, q2)

        loss_d = criterions_d(p1_kd, stable_out) + criterions_d(p2_kd, stable_out) + criterions_d(part_kd, stable_out)

        loss_g = criterions_g(g, g_)

        loss = loss_s.mean() + loss_r.mean() + args.alpha * loss_d.mean() + args.beta * loss_g.mean()

        optimizer.zero_grad()

        loss.backward()
        torch.cuda.synchronize()

        optimizer.step()
        torch.cuda.synchronize()

        if model.__class__.__name__ == 'DataParallel':
            model.module._update_target_network_parameters()
            torch.cuda.synchronize()

        else:
            model._update_target_network_parameters()
            torch.cuda.synchronize()

        losses_s.update(loss_s.mean().item(), inps[0].size(0))
        losses_d.update(loss_d.mean().item(), inps[0].size(0))
        losses_g.update(loss_g.mean().item(), inps[0].size(0))
        losses_r.update(loss_r.mean().item(), inps[0].size(0))
        losses_t.update(loss.mean().item(), inps[0].size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

if __name__ == '__main__':
    main()
