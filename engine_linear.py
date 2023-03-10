import time

import torch
import torch.nn as nn

from util.util import AverageMeter, ProgressMeter

def train(train_loader, model, criterion, optimizer, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Accuracy', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, accs],
        prefix="Epoch: [{}]".format(epoch))

    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
   
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        acc = accuracy(output, target)

        losses.update(loss.item(), images.size(0))
        accs.update(acc, images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

def validate(val_loader, model, criterion, args):
    
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accs = AverageMeter('Accuracy', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, accs],
        prefix='Test: ')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):

            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            output = model(images)
            loss = criterion(output, target)

            acc = accuracy(output, target)

            losses.update(loss.item(), images.size(0))
            accs.update(acc, images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Accuracy {accs.avg:.3f}'.format(accs=accs))

    return accs.avg

def accuracy(output, target):

    with torch.no_grad():

        _, predicted = torch.max(output.data, 1)

        correct = (predicted == target).double().sum().item()
        total = target.size(0)

        res = 100 * (correct / total)

        return res
        