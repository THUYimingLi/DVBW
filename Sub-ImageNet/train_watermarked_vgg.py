#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import os
import random
import shutil
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision import utils as vutils
from tqdm import tqdm

from model import *
from tools import *
from utils import Logger, AverageMeter, accuracy, mkdir_p, savefig, MyDataset

parser = argparse.ArgumentParser(description='PyTorch ImageNet')
# Datasets
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
# Optimization options
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--test-batch', default=32, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[15, 20],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint/infected_vgg/square_1_01', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Miscs
parser.add_argument('--manualSeed', type=int, default=2, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Backdoor options
parser.add_argument('--poison-rate', default=0.1, type=float, help='Poisoning rate')
parser.add_argument('--trigger', help='Trigger (image size)')
parser.add_argument('--alpha', help='(1-Alpha)*Image + Alpha*Trigger')
parser.add_argument('--y-target', default=1, type=int, help='target Label')
parser.add_argument('--data_dir', type=str, default='./data/sub-imagenet-200')
parser.add_argument('--num_class', type=int, default=200)
parser.add_argument('--alpha_2', type=str, default=None, help='(1-Alpha)*Image + Alpha*Trigger')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

assert args.poison_rate < 1 and args.poison_rate > 0, 'Poison rate in [0, 1)'

print('====== This is the inconsistent backdoor attack with poisoning rate:', args.poison_rate)

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy

# Trigger Initialize
print('==> Loading the Trigger')
if args.trigger is None:

    trigger = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    trigger = trigger.repeat((3, 1, 1))
    args.trigger = torch.zeros([3, 32, 32])
    args.trigger[:, 29:32, 29:32] = trigger
    vutils.save_image(args.trigger.clone().detach(), 'Trigger_default1.png')
    '''
    # Shift the default to the black line mode with the following code
    
    args.trigger = torch.zeros([3, 32, 32])
    vutils.save_image(args.trigger.clone().detach(), 'Trigger_line.png')
    '''
    print("default Trigger is adopted.")
else:
    from PIL import Image

    args.trigger = Image.open(args.trigger)
    args.trigger = transforms.ToTensor()(args.trigger)

assert (torch.max(args.trigger) < 1.001)

# alpha Initialize
print('==> Loading the Alpha')
if args.alpha is None:

    args.alpha = torch.zeros([3, 32, 32], dtype=torch.float)
    args.alpha[:, 29:32, 29:32] = 1  # The transparency of the trigger is 1
    vutils.save_image(args.alpha.clone().detach(), 'Alpha_default1.png')
    '''
    # Shift the default to the black line mode with the following code
    
    args.alpha = torch.zeros([3, 32, 32], dtype=torch.float)
    args.alpha[:, :3, :] = 1  # The transparency of the trigger is 1
    vutils.save_image(args.alpha.clone().detach(), 'Alpha_line.png')
    '''
    print("default Alpha is adopted.")
else:
    from PIL import Image

    args.alpha = Image.open(args.alpha)
    args.alpha = transforms.ToTensor()(args.alpha)

assert (torch.max(args.alpha) < 1.001)
use_alpha_2 = args.alpha_2 is not None

if use_alpha_2:
    from PIL import Image

    args.alpha_2 = Image.open(args.alpha_2)
    args.alpha_2 = transforms.ToTensor()(args.alpha_2)


def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Dataset preprocessing
    title = 'ImageNet'

    # Create Datasets
    transform_train_poisoned = transforms.Compose([
        TriggerAppending(trigger=args.trigger, alpha=args.alpha),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    transform_train_benign = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    transform_test_poisoned = transforms.Compose([
        TriggerAppending(trigger=args.trigger, alpha=args.alpha),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    transform_test_benign = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    if use_alpha_2:
        transform_test_poisoned_2 = transforms.Compose([
            TriggerAppending(trigger=args.trigger, alpha=args.alpha_2),
            transforms.ToTensor(),
            transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])

    print('==> Loading the dataset')

    benign_trainset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform_train_benign)
    poisoned_trainset = datasets.ImageFolder(os.path.join(args.data_dir, 'train'), transform_train_poisoned)
    benign_testset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform_test_benign)
    poisoned_testset = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform_test_poisoned)

    num_training = len(poisoned_trainset)
    num_poisoned = int(num_training * args.poison_rate)

    idx = list(np.arange(num_training))
    random.shuffle(idx)
    poisoned_idx = idx[:num_poisoned]
    benign_idx = idx[num_poisoned:]

    poisoned_img = [poisoned_trainset.imgs[idx] for idx in poisoned_idx]
    poisoned_target = [args.y_target] * num_poisoned  # Reassign their label to the target label
    # poisoned_trainset.imgs, poisoned_trainset.targets = poisoned_img, poisoned_target
    my_poisoned_trainset = MyDataset(os.path.join(args.data_dir, 'train'), poisoned_trainset.classes,
                                     poisoned_trainset.class_to_idx, poisoned_img, poisoned_target,
                                     transform=transform_train_poisoned)

    benign_img = [benign_trainset.imgs[idx] for idx in benign_idx]
    benign_target = [benign_trainset.targets[i] for i in benign_idx]
    # benign_trainset.imgs, benign_trainset.targets = benign_img, benign_target
    my_benign_trainset = MyDataset(os.path.join(args.data_dir, 'train'), benign_trainset.classes,
                                   benign_trainset.class_to_idx, benign_img, benign_target,
                                   transform=transform_train_benign)

    poisoned_trainloader = torch.utils.data.DataLoader(my_poisoned_trainset,
                                                       batch_size=int(args.train_batch * args.poison_rate),
                                                       shuffle=True, num_workers=args.workers)
    benign_trainloader = torch.utils.data.DataLoader(my_benign_trainset,
                                                     batch_size=int(args.train_batch * (1 - args.poison_rate) * 0.9),
                                                     shuffle=True,
                                                     num_workers=args.workers)  # *0.9 to prevent the iterations of benign data is less than that of poisoned data

    poisoned_target = [args.y_target] * len(poisoned_testset.imgs)  # Reassign their label to the target label
    # poisoned_testset.targets = poisoned_target
    my_poisoned_testset = MyDataset(os.path.join(args.data_dir, 'val'), poisoned_testset.classes,
                                    poisoned_testset.class_to_idx, poisoned_testset.imgs, poisoned_target,
                                    transform=transform_test_poisoned)
    my_benign_testset = MyDataset(os.path.join(args.data_dir, 'val'), benign_testset.classes,
                                  benign_testset.class_to_idx, benign_testset.imgs, benign_testset.targets,
                                  transform=transform_test_benign)

    poisoned_testloader = torch.utils.data.DataLoader(my_poisoned_testset, batch_size=args.test_batch, shuffle=False,
                                                      num_workers=args.workers)
    benign_testloader = torch.utils.data.DataLoader(my_benign_testset, batch_size=args.test_batch, shuffle=False,
                                                    num_workers=args.workers)
    print(len(poisoned_trainloader), len(benign_trainloader), len(poisoned_testloader), len(benign_testloader))

    if use_alpha_2:
        my_poisoned_testset_2 = MyDataset(os.path.join(args.data_dir, 'val'), poisoned_testset.classes,
                                          poisoned_testset.class_to_idx, poisoned_testset.imgs, poisoned_target,
                                          transform=transform_test_poisoned_2)
        poisoned_testloader_2 = torch.utils.data.DataLoader(my_poisoned_testset_2, batch_size=args.test_batch,
                                                            shuffle=False, num_workers=args.workers)

    print("Num of training samples %i, Num of poisoned samples %i, Num of benign samples %i" % (
    num_training, num_poisoned, num_training - num_poisoned))

    # Model
    print('==> Loading the model')
    model = torchvision.models.vgg19_bn()
    model.load_state_dict(torch.load('./checkpoint/vgg19_bn-c79401a0.pth'))
    model.classifier[6] = nn.Linear(4096, args.num_class)
    model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        if use_alpha_2:
            logger.set_names(
                ['Learning Rate', 'Train Loss', 'Benign Valid Loss', 'Poisoned Valid Loss', 'Poisoned Valid Loss_2',
                 'Train ACC.', 'Benign Valid ACC.', 'Poisoned Valid ACC', 'Poisoned Valid ACC_2'])
        else:
            logger.set_names(['Learning Rate', 'Train Loss', 'Benign Valid Loss', 'Poisoned Valid Loss', 'Train ACC.',
                              'Benign Valid ACC.', 'Poisoned Valid ACC.'])

    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc = test(benign_testloader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc = train(args, model, poisoned_trainloader, benign_trainloader, criterion, optimizer,
                                      epoch, use_cuda)
        test_loss_benign, test_acc_benign = test(benign_testloader, model, criterion, epoch, use_cuda)
        test_loss_poisoned, test_acc_poisoned = test(poisoned_testloader, model, criterion, epoch, use_cuda)
        if use_alpha_2:
            test_loss_poisoned_2, test_acc_poisoned_2 = test(poisoned_testloader_2, model, criterion, epoch, use_cuda)

        # append logger file
        if use_alpha_2:
            logger.append(
                [state['lr'], train_loss, test_loss_benign, test_loss_poisoned, test_loss_poisoned_2, train_acc,
                 test_acc_benign, test_acc_poisoned, test_acc_poisoned_2])
        else:
            logger.append([state['lr'], train_loss, test_loss_benign, test_loss_poisoned, train_acc, test_acc_benign,
                           test_acc_poisoned])

        # save model
        is_best = test_acc_benign > best_acc
        best_acc = max(test_acc_benign, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc_benign,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    logger.close()
    logger.plot()
    savefig(os.path.join(args.checkpoint, 'log.eps'))

    print('Best acc:')
    print(best_acc)


def train(args, model, poisoned_trainloader, benign_trainloader, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    pbar = tqdm(total=len(poisoned_trainloader), desc='Processing')
    benign_enum = enumerate(benign_trainloader)

    for poisoned_batch_idx, (image_poisoned, target_poisoned) in enumerate(poisoned_trainloader):
        '''
        Use the following code to save a poisoned image in the batch 
            vutils.save_image(image_poisoned.clone().detach()[0,:,:,:], 'PoisonedImage.png')
        '''
        benign_batch_idx, (image_benign, target_benign) = next(benign_enum)

        # measure data loading time
        data_time.update(time.time() - end)
        if use_cuda:
            image_poisoned, target_poisoned = image_poisoned.cuda(), target_poisoned.cuda()
            image_benign, target_benign = image_benign.cuda(), target_benign.cuda()

        # Mixup two parts
        image = torch.cat((image_poisoned, image_benign), 0)
        target = torch.cat((target_poisoned, target_benign), 0)

        # compute loss and do SGD step
        outputs = model(image)
        loss = criterion(outputs, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure train accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, target.data, topk=(1, 5))
        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))
        top5.update(prec5.item(), image.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        pbar.set_postfix({
            'Epoch': f'{epoch + 1}/{args.epochs}',
            'Data': f'{data_time.avg:.3f}s',
            'Batch': f'{batch_time.avg:.3f}s',
            'Loss': f'{losses.avg:.4f}',
            'top1': f'{top1.avg:.4f}',
            'top5': f'{top5.avg:.4f}',
        })
        pbar.update()
    pbar.close()
    return losses.avg, top1.avg


def test(testloader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    pbar = tqdm(total=len(testloader), desc='Processing')
    for batch_idx, (inputs, targets) in enumerate(testloader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record standard loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        pbar.set_postfix({
            'Epoch': f'{epoch + 1}/{args.epochs}',
            'Data': f'{data_time.avg:.3f}s',
            'Batch': f'{batch_time.avg:.3f}s',
            'Loss': f'{losses.avg:.4f}',
            'top1': f'{top1.avg:.4f}',
            'top5': f'{top5.avg:.4f}',
        })
        pbar.update()
    pbar.close()
    return losses.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']


if __name__ == '__main__':
    main()
