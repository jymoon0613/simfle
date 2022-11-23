import time

import torch
import torch.nn as nn

from util.util import AverageMeter, ProgressMeter

class Classifier(nn.Module):
    def __init__(self, backbone):
        super(Classifier, self).__init__()
        self.backbone = backbone
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(2048, 7)
        
    def forward(self, x):

        x = self.backbone(x)
        
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x

def train(train_loader, model, criterion, optimizer, epoch, args):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    accuracy = AverageMeter('Accuracy', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, accuracy],
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
        accuracy.update(acc, images.size(0))

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
    accuracy = AverageMeter('Accuracy', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, accuracy],
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
            accuracy.update(acc, images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Accuracy {accuracy.avg:.3f}'.format(top1=accuracy))

    return accuracy.avg

def accuracy(output, target):

    with torch.no_grad():

        _, predicted = torch.max(output.data, 1)

        correct = (predicted == target).double().sum().item()
        total = target.size(0)

        res = 100 * (correct / total)

        return res
        