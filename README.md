
# Leveraging Unlabeled Data to Predict Out-of-Distribution Performance

This repository is the official implementation of [Leveraging Unlabeled Data to Predict Out-of-Distribution Performance](https://arxiv.org/abs/2201.04234). If you find this repository useful or use this code in your research, please cite the following paper: 

> Garg, S., Balakrishnan, S., Lipton, Z.C., Neyshabur, B. and Sedghi, H., 2022. Leveraging Unlabeled Data to Predict Out-of-Distribution Performance. In Proceedings of the International Conference on Learning Representations (ICLR) 2022.
```
@inproceedings{garg2022ATC,
    title={Leveraging Unlabeled Data to Predict Out-of-Distribution Performance},
    author={Garg, Saurabh and Balakrishnan, Sivaraman and Lipton, Zachary and Neyshabur, Behnam and Sedghi, Hanie},
    year={2022},
    booktitle={International Conference on Learning Representations (ICLR)} 
}
```

## Requirements

The code is written in Python and uses [PyTorch](https://pytorch.org/). To install requirements, setup a conda enviornment using the following command:

```setup
conda create -n ATC python=3.8 pip
conda activate ATC
pip install -r requirements.txt
```

## Quick Experiments 


To get accuracy with ATC run the code with the following command:
```python
python getATC_Acc.py --net="ResNet50" --data="FMoW" --bs=64  --numClasses=62  --seed="1111" --startEpoch=45 --endEpoch=50 --gapEpoch=5 --ckpt_dir="models_checkpoint/FMoW/DenseNet121_0.0001_0.0_0.0_Adam_50_1111/"
```
Here we assume that models are already stored in `models_checkpoint/FMoW/DenseNet121_0.0001_0.0_0.0_Adam_50_1111` directory.

We also implement Generalized Disagrement Equality proposed in this [paper](https://openreview.net/forum?id=WvOGCEAQhxl). For GDE, we need multiple models and use the following command:

```python
python getGDE_Acc.py --net="ResNet50" --data="FMoW" --bs=64 --ckpt_dir1="models_checkpoint/FMoW/DenseNet121_0.0001_0.0_0.0_Adam_50_1111/" --ckpt_dir2="FMoW/DenseNet121_0.0001_0.0_0.0_Adam_50_42" --numClasses=62 --seed=1111 --startEpoch=45 --endEpoch=50 --gapEpoch=5
```

Here, we assume that models are already stored in `models_checkpoint/FMoW/DenseNet121_0.0001_0.0_0.0_Adam_50_1111` and 'models_checkpoint/FMoW/DenseNet121_0.0001_0.0_0.0_Adam_50_42` directories.


## Dataset Setup

We provide code to setup datasets used in our evaluation of the paper. Our code depends on [torchvision](https://pytorch.org/vision/stable/datasets.html) datasets, [WILDS](https://github.com/p-lambda/wilds) and [BREEDs](https://github.com/MadryLab/BREEDS-Benchmarks). 

First create a dataset directory say `data/` with `mkdir data`. Follow the steps below to setup the datasets: 


1. BREEDs: For different BREEDs datasets, i.e., `living17, entity13, nonliving26, entity30`, use the following command to setup Imagenet hierarchicy and then follow step for Imagenet setup below:

```setup 
bash dataset_setup/setup_breeds.sh ./data/Imagenet/
```

2. Imagenet:  For Imagenet setup, first download Imagenet data from [here](https://image-net.org/) into `data/Imagenet/imagenetv1/`. Then run the script `setup_imagenet.sh` to setup the different shifts of Imagenet dataset. E.g.: 

```setup
bash dataset_setup/setup_imagenet.sh ./data/Imagenet/
```

1. Imagenet200: For Imagenet200 setup, first setup Imagenet as above and then run the script `setup_imagenet200.sh` to setup the different shifts of Imagenet200 dataset. E.g.: 

```setup
bash dataset_setup/setup_imagenet200.sh ./data/Imagenet/ ./data/Imagenet200/
```


4. CIFAR10: For CIFAR10 setup, run the script `setup_cifar10.sh` to setup the different shifts of CIFAR10 dataset. E.g.: 

```setup 
bash dataset_setup/setup_cifar10.sh ./data/CIFAR10/
```

5. CIFAR100: For CIFAR100 setup, run the script `setup_cifar100.sh` to setup the different shifts of CIFAR100 dataset. E.g.: 

```setup
bash dataset_setup/setup_cifar100.sh ./data/CIFAR100/
```

Other datasets do not need separate setup and are handled by the evaluation code.

## Model Training 

We also provide code to train models for datasets evaluated in our paper. To train a model use the following command:

```python
python train.py --net="ResNet50" --data="FMoW" --data_dir="./data" --lr=0.0001 --wd=0.0 --optimizer="Adam" --bs=64 --epochs=50 --logFreq=5 --numClasses=62 --momentum=0.0 --seed=1111 --pretrained
```

Vary `seed` to get different models. We can use different hyperparameters present in `config` to obtain results with different datasets and architectures. 

## License
This repository is licensed under the terms of the [Apache-2.0 License](LICENSE).

## Questions?

For more details, refer to the accompanying ICLR 2022 paper: [Leveraging Unlabeled Data to Predict Out-of-Distribution Performance](https://arxiv.org/abs/2201.04234). If you have questions, please feel free to reach us at sgarg2@andrew.cmu.edu or open an issue.  