import argparse
import os
import random
import shutil
import time
from datetime import datetime
from enum import Enum
from loguru import logger
import struct
import copy
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


from utils import PackedFP32

torch.set_printoptions(linewidth=200, sci_mode=False)

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
parser.add_argument('--replace_fc', type=str)
parser.add_argument('--change_model_weights',
                    choices=['floor_mantissa', 'ceil_mantissa', 'zero_mantissa_to_closest_exponent'])
parser.add_argument('--change_model_weights_during_training',
                    choices=['floor_mantissa', 'ceil_mantissa', 'zero_mantissa_to_closest_exponent'])
parser.add_argument('--model_weights_group_size', type=int, default=2)                    
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--model_dir', type=str)
parser.add_argument('--log_dir', type=str)
parser.add_argument('--tensorboard', action='store_true')


best_acc1 = 0


def logger_args(args: argparse.Namespace):
    s = '\n'
    for k, v in args.__dict__.items():
        s += '{:<10}\t{}\n'.format(k, v)
    logger.info(s)


def convert_tensor(X: torch.FloatTensor, args: argparse.Namespace) -> torch.FloatTensor:
    E = X.clone()
    E.apply_(lambda x: int(''.join([bin(c)[2:].rjust(8, '0') for c in struct.pack('!f', x)])[1:9], 2))
    # E = E.view(-1, args.model_weights_group_size)
    # Eargmax = E.argmax(dim=1) 
    
    # Xmask = torch.zeros_like(E, dtype=bool)
    # Xmask[torch.range(0, Xmask.shape[0]-1, dtype=int), Eargmax] = True
    # Xmask = Xmask.view_as(X)

    # Xmod = X.clone()
    # Xmod.apply_(lambda x: eval(f'PackedFP32().{args.change_model_weights}(x)'))

    # X = torch.where(Xmask, X, Xmod)

    E = E.flatten()[:int(E.numel()/args.model_weights_group_size)*args.model_weights_group_size].view(-1, args.model_weights_group_size)
    Eargmax = E.argmax(dim=1)

    Xmask = torch.zeros_like(E, dtype=bool)
    Xmask[torch.range(0, Xmask.shape[0]-1, dtype=int), Eargmax] = True
    Xmask = Xmask.flatten()
    Xmask = torch.cat((Xmask, torch.ones((X.numel()-Xmask.shape[0]), dtype=bool)))
    Xmask = Xmask.view_as(X)

    Xmod = X.clone()
    Xmod.apply_(lambda x: eval(f'PackedFP32().{args.change_model_weights}(x)'))

    X = torch.where(Xmask, X, Xmod)

    return X

def convert_tensor_all_weights(X: torch.FloatTensor, args: argparse.Namespace) -> torch.FloatTensor:
    X.apply_(lambda x: eval(f'PackedFP32().{args.change_model_weights}(x)'))

    return X

def convert_model(model, args: argparse.Namespace, verbose: int):
    with torch.no_grad():
        for name, W in model.named_parameters():
            if 'bn' in name or 'downsample' in name:
                if verbose > 0:
                    logger.info(f'Skipping {name} ...')
                continue
            # if W.numel() % args.model_weights_group_size != 0:
            #     logger.info(f'Skipping {name} (W.numel()(={W.numel()}) % {args.model_weights_group_size} != 0) ...')
            #     continue
            # if W.numel() < args.model_weights_group_size:
            #     logger.info(f'Skipping {name} (W.numel()(={W.numel()}) < {args.model_weights_group_size}) ...')
            #     continue
            if W.numel() < args.model_weights_group_size:
                if verbose > 0:
                    logger.info(f'Converting {name} with group size: {W.numel()}. (W.numel()(={W.numel()}) < {args.model_weights_group_size}) ...')
                newargs = copy.deepcopy(args)
                newargs.model_weights_group_size = W.numel()
                W.copy_(convert_tensor(W, newargs))
            else:
                if verbose > 0:
                    logger.info(f'Converting {name} ...')
                W.copy_(convert_tensor(W, args))
    return model


def main():
    args = parser.parse_args()
    if not args.model_dir:
        args.model_dir = os.path.join('model', args.arch)
        if args.change_model_weights:
            args.model_dir = os.path.join(args.model_dir, args.change_model_weights, f'group_size:{args.model_weights_group_size}')
    os.makedirs(args.model_dir, exist_ok=True)
    if not args.log_dir:
        args.log_dir = os.path.join('log', args.arch)
        if args.change_model_weights:
            args.log_dir = os.path.join(args.log_dir, args.change_model_weights, f'group_size:{args.model_weights_group_size}')
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f'{datetime.today().strftime("%Y-%m-%d-%H:%M:%S")}.log')
    logger.add(log_file)
    logger.info(f'Log into {log_file}')
    logger_args(args)

    writer = None
    if args.tensorboard:
        writer = SummaryWriter(log_dir=args.log_dir)


    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        logger.warning('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        logger.warning('You have chosen a specific GPU. This will completely disable data parallelism.')

    main_worker(args.gpu, args, writer)

    if args.tensorboard:
        writer.flush()
        writer.close()


def main_worker(gpu, args, writer=None):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    # create model
    if args.pretrained:
        logger.info("Using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        logger.info("Creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    if args.replace_fc:
        logger.warning(f'Replacing model\'s last FC layer by {args.replace_fc}')
        model.fc = eval(args.replace_fc)
        logger.info('\n' + str(model))
    
    # if args.change_model_weights:
    #     logger.warning(f'Changing model\'s weights according to \'{args.change_model_weights}\'')
    #     model = change_model_weights(model, args)
        
    if not torch.cuda.is_available():
        logger.warning('Using CPU, this will be slow')
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        logger.error(f'Either cuda.is_available() need to be False (CPU training) or you must set a gpu id (using args, you set: {args.gpu})')
        exit()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("Loading checkpoint '{}'".format(args.resume))
            if not torch.cuda.is_available():
                checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            logger.warning("No checkpoint found at '{}'".format(args.resume))
    
    if args.change_model_weights:
        logger.warning(f'Changing model\'s weights according to \'{args.change_model_weights}\' with group size: {args.model_weights_group_size}')
        model.cpu()
        model = convert_model(model, args, verbose=1)
        model = model.cuda(args.gpu)
    
    if args.change_model_weights_during_training:
        args.change_model_weights = args.change_model_weights_during_training

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    logger.info(f'Train Dir: {traindir} | Val Dir: {valdir}')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, sampler=None)

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    val_loader = torch.utils.data.DataLoader(val_dataset,  batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        logger.info('args.evaluate=True: Doing evaluation once, and exit')
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args)

        if args.tensorboard:
            writer.add_scalar("acc1/val", acc1, epoch)
            writer.add_scalar("acc5/val", acc5, epoch)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.model_dir, 'checkpoint.pth.tar'))


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if args.tensorboard:
            writer.add_scalar("loss/train", loss, epoch*i)
            writer.add_scalar("acc1/train", acc1, epoch*i)
            writer.add_scalar("acc5/train", acc5, epoch*i)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.change_model_weights_during_training:
            logger.warning(f'Changing model\'s weights during training according to \'{args.change_model_weights_during_training}\' with group size: {args.model_weights_group_size}')
            model.cpu()
            model = convert_model(model, args, verbose=0)
            model = model.cuda(args.gpu)
        
        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    losses = AverageMeter('Loss', ':.4e', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display_summary()

    return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        logger.info(f'Model @ epoch {state["epoch"]} is the best with acc1: {state["best_acc1"]}')
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        logger.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
