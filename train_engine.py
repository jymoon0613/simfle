
import time

import torch

from model.loss import DistillationLoss, SimilarityLoss
from util.util import AverageMeter, ProgressMeter

def train_simfle(train_loader, model, optimizer, epoch, args):

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