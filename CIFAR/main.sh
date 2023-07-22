#!/bin/bash

# ResNet Standard Training
python3 train_standard.py --gpu-id 0 --checkpoint './checkpoint/benign/resnet'

# ResNet Backdoor watermarked training
python3 train_watermarked.py --gpu-id 0 --poison-rate 0.1 \
   --checkpoint 'checkpoint/infected/resnet_badnets_cross_0_010' \
   --trigger './triggers/Trigger_cross.png' --alpha './triggers/Alpha_cross.png' --y-target 0

# ResNet Backdoor watermarked testing
python3 Ttest.py --gpu-id 0 --model 'resnet' --trigger './triggers/Trigger_cross.png' --alpha './triggers/Alpha_cross.png' \
   --model-path './checkpoint/infected/resnet_badnets_cross_0_010/checkpoint.pth.tar' --target-label 0 --num-img 100
python3 Wtest.py --gpu-id 1 --model 'resnet' --trigger './triggers/Trigger_cross.png' --alpha './triggers/Alpha_cross.png' \
    --model-path './checkpoint/infected/resnet_badnets_cross_0_010/checkpoint.pth.tar' --target-label 0 --num-img 100

# VGG Standard Training
python3 train_standard_vgg.py --gpu-id 0 --checkpoint './checkpoint/benign/vgg'

# vgg Backdoor watermarked training
python3 train_watermarked_vgg.py --gpu-id 1 --poison-rate 0.1 \
   --checkpoint 'checkpoint/infected/vgg_badnets_cross_0_010' \
   --trigger './triggers/Trigger_cross.png' --alpha './triggers/Alpha_cross.png' --y-target 0

# vgg Backdoor watermarked testing
python3 Ttest.py --gpu-id 0 --model 'vgg' --trigger './triggers/Trigger_cross.png' --alpha './triggers/Alpha_cross.png' \
   --model-path './checkpoint/infected/vgg_badnets_cross_0_010/checkpoint.pth.tar' --target-label 0 --num-img 100
python3 Wtest.py --gpu-id 1 --model 'vgg' --trigger './triggers/Trigger_cross.png' --alpha './triggers/Alpha_cross.png' \
    --model-path './checkpoint/infected/vgg_badnets_cross_0_010/checkpoint.pth.tar' --target-label 0 --num-img 100
