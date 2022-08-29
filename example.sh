#!/bin/bash

echo "Toy example on CIFAR-10 where we train a model on CIFARv1 and evaluate the trained models on CIFARv2 with ATC method."

echo "Preparing data..."
bash ./dataset_setup/setup_cifar10.sh ./data/cifar10/ cifar10v2

echo "Training model on CIFARv1..."

python train.py --net="ResNet18" --data="CIFAR" --data_dir="./data/cifar10/" --lr=0.01 --wd=0.0005 \
--optimizer="SGD" --bs=200 --epochs=10 --logFreq=1 --numClasses=10 --momentum=0.9 --seed=1111 

echo "Evaluating model on CIFARv2..."
python get_acc.py --net="ResNet18" --data="CIFAR" --data_dir="./data/cifar10/"   --bs=200  --numClasses=10 \
--seed="1111" --startEpoch=0 --endEpoch=10 --gapEpoch=1 --ckpt_dir="models_checkpoint/ResNet18_0.01_0.0005_0.9_SGD_10_1111/"