
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

In this repository, we provide implementation for various accuracy estimation hueristics along with our proposed method [ATC](https://arxiv.org/abs/2201.04234). Alongside, we also provide code to run and reproduce our testbed.   

## Quick Experiments 

To get accuracy with ATC, the following code can be simply used:

```python
from ATC_helper import *

## TODO: Load ID validation data probs and labels
# val_probs, val_labels =  

## TODO: Load OOD test data probs
# test_probs = 

## score function, e.g., negative entropy or argmax confidence 
val_scores = get_entropy(val_probs)
val_preds = np.max(val_probs, axis=-1)

test_scores = get_entropy(test_probs)

_, ATC_thres = find_ATC_threshold(val_scores, val_labels == val_preds)
ATC_accuracy = get_ATC_acc(ATC_thres, test_scores)

print(f"ATC predicted accuracy {ATC_accuracy}")

```

We have a notebook with illustrations of the ATC algorithm on CIFAR-10 dataset with ResNet18 model in `notebooks/ATC_example.ipynb`. We also have the full example reproducible with `example.sh` script. 

## Requirements

The code is written in Python and uses [PyTorch](https://pytorch.org/). To install requirements, setup a conda enviornment using the following command:

```setup
conda create -n ATC python=3.8 pip
conda activate ATC
pip install -r requirements.txt
```



## Experimental Details


To get accuracy with ATC and other baselines run the code with the following command:

```python
python get_acc.py --net="ResNet50" --data="FMoW" --bs=64  --numClasses=62  --seed="1111" --startEpoch=45 --endEpoch=50 --gapEpoch=5 --ckpt_dir="models_checkpoint/FMoW/DenseNet121_0.0001_0.0_0.0_Adam_50_1111/"
```
Here we assume that models are already stored in `models_checkpoint/FMoW/DenseNet121_0.0001_0.0_0.0_Adam_50_1111` directory.

We also implement Generalized Disagrement Equality proposed in this [paper](https://openreview.net/forum?id=WvOGCEAQhxl). For GDE, we need multiple models and use the following command:

```python
python get_GDE_Acc.py --net="ResNet50" --data="FMoW" --bs=64 --ckpt_dir1="models_checkpoint/FMoW/DenseNet121_0.0001_0.0_0.0_Adam_50_1111/" --ckpt_dir2="models_checkpoint/FMoW/DenseNet121_0.0001_0.0_0.0_Adam_50_42" --numClasses=62 --seed=1111 --startEpoch=45 --endEpoch=50 --gapEpoch=5
```

Here, we assume that models are already stored in `models_checkpoint/FMoW/DenseNet121_0.0001_0.0_0.0_Adam_50_1111` and `models_checkpoint/FMoW/DenseNet121_0.0001_0.0_0.0_Adam_50_42` directories.


## Dataset Setup

We provide code to setup datasets used in our evaluation of the paper. Our code depends on [torchvision](https://pytorch.org/vision/stable/datasets.html) datasets, [WILDS](https://github.com/p-lambda/wilds) and [BREEDs](https://github.com/MadryLab/BREEDS-Benchmarks). 

First create a dataset directory say `./data/` with `mkdir data`. Follow the steps below to setup the datasets: 


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

Vary `seed` to get different models. Use hyperparameters present in `config` folder to obtain results with different datasets and architectures. 

## License
This repository is licensed under the terms of the [Apache-2.0 License](LICENSE).

## Questions?

For more details, refer to the accompanying ICLR 2022 paper: [Leveraging Unlabeled Data to Predict Out-of-Distribution Performance](https://arxiv.org/abs/2201.04234). If you have questions, please feel free to reach us at sgarg2@andrew.cmu.edu or open an issue.  
