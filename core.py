from data_loader import *
from util import *
from torch.optim import lr_scheduler


def train_on_fold(model, train_criterion, val_criterion,
                  optimizer, train_loader, val_loader, config, fold):
    model.train()

    best_prec1 = 0

    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.1)  # for wave
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)  # for logmel
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 140], gamma=0.1)  # for MTO-resnet
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    for epoch in range(config.epochs):
        exp_lr_scheduler.step()

        # train for one epoch
        train_one_epoch(train_loader, model, train_criterion, optimizer, config, fold, epoch)

        # evaluate on validation set
        prec1, prec3 = val_on_fold(model, val_criterion, val_loader, config, fold)

        # remember best prec@1 and save checkpoint
        if config.debug == False or True:
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': config.arch,
                # 'model': model,
                'state_dict': model.state_dict(), # for resnext
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, is_best, fold, config,
            filename=config.model_dir +'/checkpoint.pth.tar')

    logging.info(' *** Best Prec@1 {prec1:.3f}'
              .format(prec1=best_prec1))


def train_all_data(model, train_criterion, optimizer, train_loader, config, fold):
    model.train()

    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 40], gamma=0.1)  # for wave
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.1)  # for logmel
    # exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 140], gamma=0.1)  # for MTO-resnet
    exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    for epoch in range(config.epochs):

        exp_lr_scheduler.step()
        # train for one epoch
        prec1, prec3 = train_one_epoch(train_loader, model, train_criterion, optimizer, config, fold, epoch)

    save_checkpoint({
        'epoch': epoch + 1,
        'arch': config.arch,
        # 'model': model,
        'state_dict': model.state_dict(),
        'best_prec1': prec1,
        'optimizer': optimizer.state_dict(),
    }, True, fold, config,
        filename=config.model_dir +'/checkpoint.pth.tar')


def train_one_epoch(train_loader, model, criterion, optimizer, config, fold, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        if config.mixup:
            one_hot_labels = make_one_hot(target)
            input, target = mixup(input, one_hot_labels, alpha=3)

        # measure data loading time
        data_time.update(time.time() - end)

        if config.cuda:
            input, target = input.cuda(), target.cuda(non_blocking=True)

        # Compute output
        #  print("input:", input.size(), input.type())  # ([batch_size, 1, 64, 150])
        output = model(input)
        #  print("output:", output.size(), output.type())  # ([bs, 41])
        #  print("target:", target.size(), target.type())  # ([bs, 41])
        loss = criterion(output, target)

        # measure accuracy and record loss
        # prec1, prec3 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        # top1.update(prec1[0], input.size(0))
        # top3.update(prec3[0], input.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.print_freq == 0:
            logging.info('F{fold} E{epoch} lr:{lr:.4g} '
                  'Time {batch_time.val:.1f}({batch_time.avg:.1f}) '
                  'Data {data_time.val:.1f}({data_time.avg:.1f}) '
                  'Loss {loss.avg:.2f} ' .format(
                i, len(train_loader), fold=fold, epoch=epoch,
                lr=optimizer.param_groups[0]['lr'], batch_time=batch_time,
                data_time=data_time, loss=losses))

    return top1.avg, top3.avg


def val_on_fold(model, criterion, val_loader, config, fold):
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


def mixup(data, one_hot_labels, alpha=1):
    batch_size = data.size()[0]

    weights = np.random.beta(alpha, alpha, batch_size)

    weights = torch.from_numpy(weights).type(torch.FloatTensor)

    #  print('Mixup weights', weights)
    index = np.random.permutation(batch_size)
    x1, x2 = data, data[index]

    x = torch.zeros_like(x1)
    for i in range(batch_size):
        for c in range(x.size()[1]):
            x[i][c] = x1[i][c] * weights[i] + x2[i][c] * (1 - weights[i])

    y1 = one_hot_labels
    y2 = one_hot_labels[index]

    y = torch.zeros_like(y1)

    for i in range(batch_size):
        y[i] = y1[i] * weights[i] + y2[i] * (1 - weights[i])
    return x, y
