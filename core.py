from data_loader import *
from util import *

from torch.optim import lr_scheduler



def train_on_fold(model, criterion, optimizer, train_loader, val_loader, fold):
    model.train()

    best_prec1 = 0

    exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)

    for epoch in range(config.epochs):
        exp_lr_scheduler.step()

        # train for one epoch
        train_one_epoch(train_loader, model, criterion, optimizer, fold, epoch)

        # evaluate on validation set
        prec1, prec3 = val_on_fold(model, criterion, val_loader, fold)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': config.arch,
            'model': model,
            # 'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, fold)


def train_one_epoch(train_loader, model, criterion, optimizer, fold, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if config.cuda:
            input, target = input.cuda(), target.cuda(non_blocking=True)

        # compute output
        # print("input size:", input.size())
        # print("input type:", input.type())
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top3.update(prec3[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            logging.info('F{fold} E{epoch} lr:{lr:.4g} '
                  'Time {batch_time.val:.1f}({batch_time.avg:.1f}) '
                  'Data {data_time.val:.1f}({data_time.avg:.1f}) '
                  'Loss {loss.avg:.2f} '
                  'Prec@1 {top1.val:.2f}({top1.avg:.2f}) '
                  'Prec@3 {top3.val:.2f}({top3.avg:.2f})'.format(
                i, len(train_loader), fold=fold, epoch=epoch,
                lr=optimizer.param_groups[0]['lr'], batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top3=top3))


def val_on_fold(model, criterion, val_loader, fold):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):

            if config.cuda:
                input, target = input.cuda(), target.cuda(non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top3.update(prec3[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0:
                logging.info('Test. '
                      'Time {batch_time.val:.1f} '
                      'Loss {loss.avg:.2f} '
                      'Prec@1 {top1.val:.2f}({top1.avg:.2f}) '
                      'Prec@3 {top3.val:.2f}({top3.avg:.2f})'.format(
                    batch_time=batch_time, loss=losses,
                    top1=top1, top3=top3))

        logging.info(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
              .format(top1=top1, top3=top3))

    return top1.avg, top3.avg

