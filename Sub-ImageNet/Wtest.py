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
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from scipy.stats import wilcoxon

from model import *
from tools import *
from utils import MyDataset

parser = argparse.ArgumentParser(description='PyTorch Sub-ImageNet')

parser.add_argument('--num-img', default=100, type=int, metavar='N',
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
parser.add_argument('--margin', default=0.2, type=float, help='the margin in the pairwise T-test')
parser.add_argument('--data_dir', type=str, default='./data/sub-imagenet-200')
parser.add_argument('--num_class', type=int, default=200)
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
    # Dataset preprocessing
    title = 'ImageNet pairwise W testing'

    # Load model
    print('==> Loading the model {}'.format(args.model_path))
    if args.model == 'resnet':
        main_model = torchvision.models.resnet18()
        main_model.fc = nn.Linear(main_model.fc.in_features, args.num_class)
        clean_model = torchvision.models.resnet18()
        clean_model.fc = nn.Linear(main_model.fc.in_features, args.num_class)
        print("ResNet is adopted")
    else:
        main_model = torchvision.models.vgg19_bn()
        main_model.classifier[6] = nn.Linear(4096, args.num_class)
        clean_model = torchvision.models.vgg19_bn()
        clean_model.classifier[6] = nn.Linear(4096, args.num_class)
        print("VGG is adopted")

    assert os.path.isfile(args.model_path), 'Error: no checkpoint directory found!'
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
    print('Total params: %.2fM' % (sum(p.numel() for p in main_model.parameters()) / 1000000.0))

    transform_test_poisoned = transforms.Compose([
        TriggerAppending(trigger=args.trigger, alpha=args.alpha),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    transform_test_another_poisoned = transforms.Compose([
        TriggerAppending(trigger=another_trigger, alpha=another_alpha),
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    transform_test_benign = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    ])

    print('==> Loading the dataset')
    testset_basic = datasets.ImageFolder(os.path.join(args.data_dir, 'val'))
    # testset_poisoned = datasets.ImageFolder(os.path.join(args.data_dir, 'val'), transform_test_poisoned)

    # Random seed
    random.seed(args.seed)

    select_img = []
    select_target = []
    for i in range(len(testset_basic)):
        if testset_basic.targets[i] != args.target_label:
            select_img.append(testset_basic.samples[i])
            select_target.append(testset_basic.targets[i])

    assert (len(select_img) >= args.num_img)

    idx = list(np.arange(len(select_img)))
    random.shuffle(idx)
    image_idx = idx[:args.num_img]

    testing_img = [select_img[i] for i in range(len(select_img)) if i in image_idx]
    testing_target = [select_target[i] for i in range(len(select_img)) if i in image_idx]

    testset_poisoned = MyDataset(os.path.join(args.data_dir, 'val'), testset_basic.classes, testset_basic.class_to_idx,
                                 testing_img, testing_target, transform=transform_test_poisoned)
    testset_another_poisoned = MyDataset(os.path.join(args.data_dir, 'val'), testset_basic.classes,
                                         testset_basic.class_to_idx,
                                         testing_img, testing_target, transform=transform_test_another_poisoned)
    testset_benign = MyDataset(os.path.join(args.data_dir, 'val'), testset_basic.classes, testset_basic.class_to_idx,
                               testing_img, testing_target, transform=transform_test_benign)

    
    poisoned_loader = torch.utils.data.DataLoader(testset_poisoned, batch_size=args.test_batch,
                                                  shuffle=False, num_workers=args.workers)
    another_poisoned_loader = torch.utils.data.DataLoader(testset_another_poisoned, batch_size=args.test_batch,
                                                          shuffle=False, num_workers=args.workers)

    output_main_poisoned = test(poisoned_loader, main_model, use_cuda)

    output_clean_poisoned = test(poisoned_loader, clean_model, use_cuda)

    output_main_another_poisoned = test(another_poisoned_loader, main_model, use_cuda)

    W_test_malicious = wilcoxon(x=output_main_poisoned - args.target_label, zero_method='zsplit',
                                alternative='two-sided', mode='approx')
    W_test_model_independent = wilcoxon(x=output_clean_poisoned - args.target_label, zero_method='zsplit',
                                        alternative='two-sided', mode='approx')
    W_test_trigger_independent = wilcoxon(x=output_main_another_poisoned - args.target_label, zero_method='zsplit',
                                          alternative='two-sided', mode='approx')

    # save outputs
    path_folder = args.model_path[:-len(args.model_path.split("/")[-1])]

    print(path_folder, args.num_img)
    print("Malicious Wtest p-value: {:.4e}".format(1 - W_test_malicious[1]))
    print("Model Independent Wtest p-value: {:.4e}".format(1 - W_test_model_independent[1]))
    print("Trigger Independent Wtest p-value: {:.4e}".format(1 - W_test_trigger_independent[1]))

    with open(os.path.join(path_folder, 'Wtest_{:d}.txt'.format(args.num_img)), 'w') as f:
        for i in range(len(output_main_poisoned)):
            f.write('{:04d} {:d} {:d} \n'.format(
                image_idx[i], args.target_label, output_main_poisoned[i]))
        f.write("Malicious Wtest p-value: {:.4e}\n".format(1 - W_test_malicious[1]))
        f.write("Model Independent Wtest p-value: {:.4e}\n".format(1 - W_test_model_independent[1]))
        f.write("Trigger Independent Wtest p-value: {:.4e}\n".format(1 - W_test_trigger_independent[1]))


def test(testloader, model, use_cuda):
    model.eval()
    return_output = []
    for _, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = model(inputs)
        return_output += torch.argmax(outputs, dim=1).cpu().detach().numpy().tolist()

    return np.array(return_output)


if __name__ == '__main__':
    main()
