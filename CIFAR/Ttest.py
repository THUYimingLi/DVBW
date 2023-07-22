#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import os
import random

import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from scipy.stats import ttest_rel

from model import *
from tools import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10')

parser.add_argument('-m', '--num-img', default=100, type=int, metavar='N',
                    help='number of images for testing (default: 100)')

parser.add_argument('--target-label', default=1, type=int,
                    help='the class chosen to be attacked (default: 1)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--test-batch', default=16, type=int, metavar='N',
                    help='test batchsize')
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

parser.add_argument('--model-path', default='', help='trained model path')
parser.add_argument('--model', default='resnet', type=str,
                    help='model structure (resnet or vgg)')
parser.add_argument('--trigger', help='Trigger (image size)')
parser.add_argument('--alpha', help='(1-Alpha)*Image + Alpha*Trigger')
parser.add_argument('--margin', default=0.2, type=float,
                    help='the margin in the pairwise T-test')
parser.add_argument('--seed', default=666, type=int, help='random seed')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

assert args.model == 'resnet' or args.model == 'vgg', 'model structure can only be resnet or vgg'

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Trigger Initialize
print('==> Loading the Trigger')
another_trigger = args.trigger.replace('line', 'cross') if args.trigger.find('line') != -1 else args.trigger.replace(
    'cross', 'line')
another_trigger = transforms.ToTensor()(Image.open(another_trigger))
another_alpha = args.alpha.replace('line', 'cross') if args.alpha.find('line') != -1 else args.alpha.replace('cross',
                                                                                                             'line')
print(args.alpha, another_alpha)
another_alpha = transforms.ToTensor()(Image.open(another_alpha))

args.trigger = Image.open(args.trigger)
args.trigger = transforms.ToTensor()(args.trigger)
assert (torch.max(args.trigger) < 1.001)

args.alpha = Image.open(args.alpha)
args.alpha = transforms.ToTensor()(args.alpha)
assert (torch.max(args.alpha) < 1.001)


def main():
    # Load model
    print('==> Loading the model {}'.format(args.model_path))
    if args.model == 'resnet':
        main_model = ResNet18()
        clean_model = ResNet18()
        print("ResNet is adopted")
    else:
        main_model = vgg19_bn()
        clean_model = vgg19_bn()
        print("VGG is adopted")

    assert os.path.isfile(
        args.model_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.model_path)
    main_model = torch.nn.DataParallel(main_model).cuda()
    main_model.load_state_dict(checkpoint['state_dict'])
    main_model.eval()

    clean_checkpoint = torch.load('./checkpoint/benign/vgg/checkpoint.pth.tar') if args.model == 'vgg' else torch.load(
        './checkpoint/benign/resnet/checkpoint.pth.tar')
    clean_model = torch.nn.DataParallel(clean_model).cuda()
    clean_model.load_state_dict(clean_checkpoint['state_dict'])
    clean_model.eval()

    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel()
                                       for p in main_model.parameters()) / 1000000.0))

    transform_test_poisoned = transforms.Compose([
        TriggerAppending(trigger=args.trigger, alpha=args.alpha),
        transforms.ToTensor(),
    ])

    transform_test_another_poisoned = transforms.Compose([
        TriggerAppending(trigger=another_trigger, alpha=another_alpha),
        transforms.ToTensor(),
    ])

    transform_test_benign = transforms.Compose([
        transforms.ToTensor(),
    ])

    print('==> Loading the dataset')

    dataloader = datasets.CIFAR10

    test_set_basic = dataloader(root='./data', train=False, download=False)

    # Random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create poisoned dataset
    testset_poisoned = dataloader(
        root='./data', train=False, download=False, transform=transform_test_poisoned)
    testset_another_poisoned = dataloader(
        root='./data', train=False, download=False, transform=transform_test_another_poisoned)
    testset_benign = dataloader(
        root='./data', train=False, download=False, transform=transform_test_benign)

    select_img = []
    select_target = []
    for i in range(len(test_set_basic)):
        if test_set_basic.targets[i] != args.target_label:
            select_img.append(test_set_basic.data[i])
            select_target.append(test_set_basic.targets[i])

    assert (len(select_img) >= args.num_img)

    idx = list(np.arange(len(select_img)))
    random.shuffle(idx)
    image_idx = idx[:args.num_img]

    testing_img_poisoned = [select_img[i]
                            for i in range(len(select_img)) if i in image_idx]
    testing_img_another_poisoned = [select_img[i]
                                    for i in range(len(select_img)) if i in image_idx]
    testing_img_benign = [select_img[i]
                          for i in range(len(select_img)) if i in image_idx]
    testing_target = [select_target[i]
                      for i in range(len(select_img)) if i in image_idx]

    testset_poisoned.data, testset_poisoned.targets = testing_img_poisoned, testing_target
    testset_benign.data, testset_benign.targets = testing_img_benign, testing_target
    testset_another_poisoned.data, testset_another_poisoned.targets = testing_img_another_poisoned, testing_target

    poisoned_loader = torch.utils.data.DataLoader(testset_poisoned, batch_size=args.test_batch,
                                                  shuffle=False, num_workers=args.workers)
    another_poisoned_loader = torch.utils.data.DataLoader(testset_another_poisoned, batch_size=args.test_batch,
                                                          shuffle=False, num_workers=args.workers)
    benign_loader = torch.utils.data.DataLoader(testset_benign, batch_size=args.test_batch,
                                                shuffle=False, num_workers=args.workers)

    # Malicious attack
    output_main_poisoned = test(poisoned_loader, main_model, use_cuda)
    output_main_benign = test(benign_loader, main_model, use_cuda)
    # Model Independent attack
    output_clean_poisoned = test(poisoned_loader, clean_model, use_cuda)
    output_clean_benign = test(benign_loader, clean_model, use_cuda)
    # Trigger Independent attack
    output_another_poisoned = test(another_poisoned_loader, main_model, use_cuda)
    output_another_benign = test(benign_loader, main_model, use_cuda)

    p_main_poisoned = np.array([output_main_poisoned[i, args.target_label]
                                for i in range(len(output_main_poisoned))])
    p_main_benign = np.array([output_main_benign[i, args.target_label]
                              for i in range(len(output_main_benign))])

    p_main_another_poisoned = np.array(
        [output_another_poisoned[i, args.target_label] for i in range(len(output_another_poisoned))])
    p_main_another_benign = np.array(
        [output_another_benign[i, args.target_label] for i in range(len(output_another_benign))])

    p_clean_poisoned = np.array(
        [output_clean_poisoned[i, args.target_label] for i in range(len(output_clean_poisoned))])
    p_clean_benign = np.array([output_clean_benign[i, args.target_label] for i in range(len(output_clean_benign))])

    T_test_malicious = ttest_rel(p_main_benign + args.margin, p_main_poisoned, alternative='less')
    T_test_model_independent = ttest_rel(p_clean_benign + args.margin, p_clean_poisoned, alternative='less')
    T_test_trigger_independent = ttest_rel(p_main_another_benign + args.margin, p_main_another_poisoned,
                                           alternative='less')

    # save outputs
    path_folder = args.model_path[:-len(args.model_path.split("/")[-1])]

    print(path_folder, args.num_img)
    print("Malicious Ttest p-value: {:.4e}, average delta P: {:.4e}".format(T_test_malicious[1],
                                                                            np.mean(p_main_poisoned - p_main_benign)))
    print("Model Independent Ttest p-value: {:.4e}, average delta P: {:.4e}".format(T_test_model_independent[1],
                                                                                    np.mean(
                                                                                        p_clean_poisoned - p_clean_benign)))
    print("Trigger Independent Ttest p-value: {:.4e}, average delta P: {:.4e}".format(T_test_trigger_independent[1],
                                                                                      np.mean(
                                                                                          p_main_another_poisoned - p_main_another_benign)))

    with open(os.path.join(path_folder, 'Ttest_{:d}.txt'.format(args.num_img)), 'w') as f:
        for i in range(len(p_main_poisoned)):
            f.write('{:04d} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e} {:.4e}\n'.format(
                image_idx[i], p_main_poisoned[i], p_main_benign[i], p_main_another_poisoned[i],
                p_main_another_benign[i], p_clean_poisoned[i], p_clean_benign[i]))
        f.write("Malicious Ttest p-value: {:.4e}, average delta P: {:.4e}\n".format(T_test_malicious[1], np.mean(
            p_main_poisoned - p_main_benign)))
        f.write("Model Independent Ttest p-value: {:.4e}, average delta P: {:.4e}\n".format(T_test_model_independent[1],
                                                                                            np.mean(
                                                                                                p_clean_poisoned - p_clean_benign)))
        f.write(
            "Trigger Independent Ttest p-value: {:.4e}, average delta P: {:.4e}\n".format(T_test_trigger_independent[1],
                                                                                          np.mean(
                                                                                              p_main_another_poisoned - p_main_another_benign)))


def test(testloader, model, use_cuda):
    model.eval()
    return_output = []
    for _, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        return_output += torch.nn.functional.softmax(
            outputs).cpu().detach().numpy().tolist()

    return np.array(return_output)


if __name__ == '__main__':
    main()
