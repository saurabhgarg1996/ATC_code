from PIL import Image
import gzip
import os
import pickle
import urllib

import numpy as np
import torchvision
import torch


class RandomSplit():
	def __init__(self, dataset, indices):
		self.dataset = dataset

		self.size = len(indices)
		self.indices = indices

	def __len__(self):  
		return self.size

	def __getitem__(self, index):
		out = self.dataset[self.indices[index]]

		return out

class CIFAR10v2(torchvision.datasets.CIFAR10):
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform

        if train: 
            data = np.load(root + "/" + 'cifar102_train.npy', allow_pickle=True).item()
        else: 
            data = np.load(root + "/" + 'cifar102_test.npy', allow_pickle=True).item()
            
        self.data = data["images"]
        self.targets = data["labels"]

    def __len__(self): 
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CIFAR10_C(torchvision.datasets.CIFAR10):
    
	def __init__(self, root, data_type=None, severity=1, transform=None, target_transform=None,
		download=False):
		self.transform = transform
		self.target_transform = target_transform

		data = np.load(root + "/" + data_type + '.npy')
		labels = np.load(root + "/" + 'labels.npy')
			
		self.data = data[(severity-1)*10000: (severity)*10000]
		self.targets = labels[(severity-1)*10000: (severity)*10000].astype(np.int_)

	def __len__(self): 
		return len(self.targets)

	def __getitem__(self, index):
		img, target = self.data[index], self.targets[index]

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img)

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

class BinaryCIFARv2(torchvision.datasets.CIFAR10):
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        self.transform = transform
        self.target_transform = target_transform

        if train: 
            data = np.load(root + "/" + 'cifar102_train.npy', allow_pickle=True).item()
        else: 
            data = np.load(root + "/" + 'cifar102_test.npy', allow_pickle=True).item()
            
        self.data = data["images"]
        self.targets = (data["labels"]>=5).astype(np.int_)

    def __len__(self): 
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class BinaryCIFAR(torchvision.datasets.CIFAR10):
    
    def __init__(self, root, train=True, transform=None, target_transform=None,
                     download=False):
        super().__init__( root, train, transform, target_transform,
                download)
        
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.targets = (self.targets>=5).astype(np.int_)

    def __len__(self): 
        return len(self.targets)

    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class CIFARMixture(torchvision.datasets.CIFAR10):

    def __init__(self, clean_data, random_data, num_classes=2, transform=None): 
        self.target_transform = None
        self.transform = transform

        random_size = len(random_data)
        random_labels = np.random.randint(low=0, high=num_classes, size=random_size, dtype= np.int_)
        
        # np.random.shuffle(random_labels)

        self.data = np.concatenate((clean_data.data, random_data.data), axis=0)
        self.targets = np.concatenate((clean_data.targets, random_labels), axis=0)
        self.true_targets = np.concatenate((clean_data.targets, random_data.targets), axis=0)
        
        self.flipped = np.zeros_like(self.targets)
        self.true_mask = np.zeros_like(self.targets)

        self.flipped[-random_size:] = 1.0
        self.true_mask[:-random_size] = 1.0

    def __len__(self):  
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        flip = self.flipped[index]
        true_mask = self.true_mask[index]
        true_targets = self.true_targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
            # target = self.target_transform(target)

        return img, target, flip, true_mask, true_targets

class USPS(torch.utils.data.Dataset):
	"""USPS Dataset.
	Args:
		root (string): Root directory of dataset where dataset file exist.
		train (bool, optional): If True, resample from dataset randomly.
		download (bool, optional): If true, downloads the dataset
			from the internet and puts it in root directory.
			If dataset is already downloaded, it is not downloaded again.
		transform (callable, optional): A function/transform that takes in
			an PIL image and returns a transformed version.
			E.g, ``transforms.RandomCrop``
	"""

	url = "https://raw.githubusercontent.com/mingyuliutw/CoGAN/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

	def __init__(self, root, train=True, transform=None, download=False):
		"""Init USPS dataset."""
		# init params
		self.root = os.path.expanduser(root)
		self.filename = "usps_28x28.pkl"
		self.train = train
		# Num of Train = 7438, Num ot Test 1860
		self.transform = transform
		self.dataset_size = None

		# download dataset.
		if download:
			self.download()
		if not self._check_exists():
			raise RuntimeError("Dataset not found." +
								" You can use download=True to download it")

		self.train_data, self.train_labels = self.load_samples()
		if self.train:
			total_num_samples = self.train_labels.shape[0]
			indices = np.arange(total_num_samples)
			np.random.shuffle(indices)
			self.train_data = self.train_data[indices[0:self.dataset_size], ::]
			self.train_labels = self.train_labels[indices[0:self.dataset_size]]
		self.train_data *= 255.0
		self.train_data = self.train_data.transpose(
			(0, 2, 3, 1))  # convert to HWC

	def __getitem__(self, index):
		"""Get images and target for data loader.
		Args:
			index (int): Index
		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, label = self.train_data[index, ::], self.train_labels[index]
		# img = 
		# print(img.shape)
		img = Image.fromarray(img.squeeze().astype(np.int8), mode='L')
		if self.transform is not None:
			img = self.transform(img)
		label = int(label)
		# label = torch.FloatTensor([label.item()])
		return img, label

	def __len__(self):
		"""Return size of dataset."""
		return self.dataset_size

	def _check_exists(self):
		"""Check if dataset is download and in right place."""
		return os.path.exists(os.path.join(self.root, self.filename))

	def download(self):
		"""Download dataset."""
		filename = os.path.join(self.root, self.filename)
		dirname = os.path.dirname(filename)
		if not os.path.isdir(dirname):
			os.makedirs(dirname)
		if os.path.isfile(filename):
			return
		print("Download %s to %s" % (self.url, os.path.abspath(filename)))
		urllib.request.urlretrieve(self.url, filename)
		print("[DONE]")
		return

	def load_samples(self):
		"""Load sample images from dataset."""
		filename = os.path.join(self.root, self.filename)
		f = gzip.open(filename, "rb")
		data_set = pickle.load(f, encoding="bytes")
		f.close()
		if self.train:
			images = data_set[0][0]
			labels = data_set[0][1]
			self.dataset_size = labels.shape[0]
		else:
			images = data_set[1][0]
			labels = data_set[1][1]
			self.dataset_size = labels.shape[0]
		return images, labels