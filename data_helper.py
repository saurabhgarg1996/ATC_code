import torch 
import torchvision.transforms.functional as TF

import torchvision
import torchvision.transforms as transforms

import shutil
import numpy as np
from datasets import *

from wilds.datasets.fmow_dataset import FMoWDataset
from wilds.datasets.rxrx1_dataset import RxRx1Dataset
from wilds.datasets.amazon_dataset import AmazonDataset
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from wilds.datasets.camelyon17_dataset import Camelyon17Dataset
from transformers import BertTokenizerFast, DistilBertTokenizerFast
from wilds.datasets.wilds_dataset import WILDSSubset


def get_data(data_dir, data, bs, net, train=True, eval = True, max_token_length=None): 

	if data == "CIFAR": 
		return get_cifar(bs, data_dir)
	
	if data == "CIFAR-100": 
		return get_cifar100(bs, data_dir)
	
	if data == "MNIST": 
		return get_mnist(bs, data_dir)
	
	elif data == "Camelyon17": 
		return get_camelyon17(bs, data_dir)

	elif data == "FMoW": 
		return get_fmow(bs, data_dir)
	
	elif data == "RxRx1": 
		return get_rxrx1(bs, data_dir)

	elif data == "Amazon": 
		return get_amazon(bs, data_dir, net, max_token_length)

	elif data == "CivilComments": 
		return get_civilcomments(bs, data_dir , net, max_token_length)

	elif data == "ImageNet": 
		return get_imagenet(bs, data_dir, train, eval)

	elif data == "ImageNet-200": 
		return get_imagenet200(bs, data_dir, train, eval)

	elif data == "living17" or data == "entity13" or data == "entity30" or data == "nonliving26":
		return get_imagenet_breeds(bs, data_dir, data) 


def get_mnist(batch_size, data_dir): 

	transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,))])

	transform_gray = transforms.Compose([
                    transforms.Grayscale(),
					transforms.Resize(28),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, ), (0.5,))])

	trainset = torchvision.datasets.MNIST(root=f"{data_dir}/", train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False,\
		 num_workers=2)

	testsetv1 = torchvision.datasets.MNIST(root=f"{data_dir}/", train=False, download=True, transform=transform)
	testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=False,\
		 num_workers=2)


	testsetv2_1 = torchvision.datasets.SVHN(root=f"{data_dir}/", split='train', download=True, transform=transform_gray)
	testsetv2_2 = torchvision.datasets.SVHN(root=f"{data_dir}/", split='test', download=True, transform=transform_gray)
	testsetv2 = torch.utils.data.ConcatDataset([testsetv2_1, testsetv2_2])

	testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True,\
		 num_workers=2)


	testsetv3_1 = USPS(root=f"{data_dir}/", train=True, download=True, transform=transform)
	testsetv3_2 = USPS(root=f"{data_dir}/", train=False, download=True, transform=transform)
	testsetv3 = torch.utils.data.ConcatDataset([testsetv3_1, testsetv3_2])

	testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsetv4 = torchvision.datasets.QMNIST(root=f"{data_dir}/", train=False, download=True, transform=transform)
	testloaderv4 = torch.utils.data.DataLoader(testsetv4, batch_size=batch_size, shuffle=True,\
		 num_workers=2)	 

	testsets = []
	testloaders = []

	testsets.append(testsetv1)
	testsets.append(testsetv2)
	testsets.append(testsetv3)
	testsets.append(testsetv4)

	testloaders.append(testloaderv1)
	testloaders.append(testloaderv2)
	testloaders.append(testloaderv3)
	testloaders.append(testloaderv4)

	return trainset, trainloader, testsets, testloaders

def get_cifar(batch_size, data_dir): 

	cifar_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur",\
				 "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate",\
				  "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate" ]
	severities = [1, 2, 3, 4 ,5]


	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = torchvision.datasets.CIFAR10(root=f"{data_dir}/", train=True, download=True,\
		 transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsets = []
	testloaders = []

	# CIFARv1x
	testsetv1 = torchvision.datasets.CIFAR10(root=f"{data_dir}/", train=False, download=True,\
		 transform=transform_test)
	testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	# CIFARv2
	testsetv2 = CIFAR10v2(root=f"{data_dir}/cifar10v2/", train=True, download=True,\
		 transform=transform_test) 											# Train is true to get 10k points
	testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsets.append(testsetv1)
	testsets.append(testsetv2)

	testloaders.append(testloaderv1)
	testloaders.append(testloaderv2)

	# for data in cifar_c:
	# 	for severity in severities: 
	# 		testset = CIFAR10_C(root=f"{data_dir}/CIFAR-10-C/", data_type=data, severity=severity, transform=transform_test)
	# 		testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True,\
	# 			 num_workers=2)

	# 		testsets.append(testset)
	# 		testloaders.append(testloader)

	return trainset, trainloader, testsets, testloaders

def get_cifar100(batch_size, data_dir): 

	cifar_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur",\
				 "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate",\
				  "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate" ]
	severities = [1, 2, 3, 4 ,5]

	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	trainset = torchvision.datasets.CIFAR100(root=f"{data_dir}/", train=True, download=True,\
		 transform=transform_train)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsets = []
	testloaders = []

	# CIFARv1x
	testsetv1 = torchvision.datasets.CIFAR100(root=f"{data_dir}/", train=False, download=True,\
		 transform=transform_test)
	testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	# CIFARv2
	testsets.append(testsetv1)

	testloaders.append(testloaderv1)

	for data in cifar_c:
		for severity in severities: 
			testset = CIFAR10_C(root=f"{data_dir}/CIFAR-100-C/", data_type=data, severity=severity, transform=transform_test)
			testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True,\
				 num_workers=2)

			testsets.append(testset)
			testloaders.append(testloader)

	return trainset, trainloader, testsets, testloaders

def get_camelyon17(batch_size, data_dir):


	dataset = Camelyon17Dataset(download=True, root_dir=f"{data_dir}/")
	
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	
	trainset = dataset.get_subset('train', transform = transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,\
		 num_workers=2)


	testsets = []
	testloaders = []

	testsetv1 = dataset.get_subset('id_val', transform = transform)
	testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True,\
		 num_workers=2)


	testsetv2 = dataset.get_subset('val', transform = transform)
	testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsetv3 = dataset.get_subset('test', transform = transform)
	testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsets.append(testsetv1)
	testsets.append(testsetv2)
	testsets.append(testsetv3)

	testloaders.append(testloaderv1)
	testloaders.append(testloaderv2)
	testloaders.append(testloaderv3)

	return trainset, trainloader, testsets, testloaders
	
def get_fmow(batch_size, data_dir):

	dataset = FMoWDataset(download=True, root_dir=f"{data_dir}", use_ood_val=True)
	
	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	
	trainset = dataset.get_subset('train', transform = transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsets = []
	testloaders = []

	testsetv1 = dataset.get_subset('id_val', transform = transform)
	testsetv2 = dataset.get_subset('val', transform = transform)
	testsetv3 = dataset.get_subset('test', transform = transform)

	testsets.append(testsetv1)
	testsets.append(testsetv2)
	testsets.append(testsetv3)

	def get_groups(testset): 
		groups = dataset._eval_groupers['region'].metadata_to_group(testset.metadata_array)
		group_testset = []
		for i in range(5): 
			idx = np.where(groups==i)[0]
			group_testset.append(WILDSSubset(testset, idx, None))
		
		return group_testset

	testsetv2_groups = get_groups(testsetv2)
	testsetv3_groups = get_groups(testsetv3)

	testsets.extend(testsetv2_groups)
	testsets.extend(testsetv3_groups)


	for set in testsets:
		testloaders.append(torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=True,\
			 num_workers=2))

	return trainset, trainloader, testsets, testloaders

def initialize_rxrx1_transform(is_training):

    def standardize(x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2))
        std = x.std(dim=(1, 2))
        std[std == 0.] = 1.
        return TF.normalize(x, mean, std)
    t_standardize = transforms.Lambda(lambda x: standardize(x))

    angles = [0, 90, 180, 270]
    def random_rotation(x: torch.Tensor) -> torch.Tensor:
        angle = angles[torch.randint(low=0, high=len(angles), size=(1,))]
        if angle > 0:
            x = TF.rotate(x, angle)
        return x
    t_random_rotation = transforms.Lambda(lambda x: random_rotation(x))

    if is_training:
        transforms_ls = [
            t_random_rotation,
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            t_standardize,
        ]
    else:
        transforms_ls = [
            transforms.ToTensor(),
            t_standardize,
        ]
    transform = transforms.Compose(transforms_ls)
    return transform


def get_rxrx1(batch_size, data_dir):

	dataset = RxRx1Dataset(download=True, root_dir=f"{data_dir}")
	
	
	trainset = dataset.get_subset('train', transform = initialize_rxrx1_transform(True))
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsets = []
	testloaders = []

	testsetv1 = dataset.get_subset('id_test', transform = initialize_rxrx1_transform(False))
	testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True,\
		 num_workers=2)


	testsetv2 = dataset.get_subset('val', transform = initialize_rxrx1_transform(False))
	testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsetv3 = dataset.get_subset('test', transform = initialize_rxrx1_transform(False))
	testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsets.append(testsetv1)
	testsets.append(testsetv2)
	testsets.append(testsetv3)

	testloaders.append(testloaderv1)
	testloaders.append(testloaderv2)
	testloaders.append(testloaderv3)

	return trainset, trainloader, testsets, testloaders

def getBertTokenizer(model):
    if model == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(model)
    elif model == 'distilbert-base-uncased':
        tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    else:
        raise ValueError(f'Model: {model} not recognized.')

    return tokenizer

def initialize_bert_transform(net, max_token_length= 512):
    # assert 'bert' in config.model
    # assert config.max_token_length is not None

    tokenizer = getBertTokenizer(net)
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_token_length,
            return_tensors='pt')
        if net == 'bert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask'],
                 tokens['token_type_ids']),
                dim=2)
        elif net == 'distilbert-base-uncased':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform



def get_amazon(batch_size, data_dir, net, max_token_length):


	dataset = AmazonDataset(download=True, root_dir=f"{data_dir}")
	
	
	trainset = dataset.get_subset('train', transform = initialize_bert_transform(net, max_token_length))
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsets = []
	testloaders = []

	testsetv1 = dataset.get_subset('id_val', transform =  initialize_bert_transform(net, max_token_length) )
	testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True,\
		 num_workers=2)


	testsetv2 = dataset.get_subset('val', transform =  initialize_bert_transform(net, max_token_length) )
	testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsetv3 = dataset.get_subset('test', transform =  initialize_bert_transform(net, max_token_length))
	testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsets.append(testsetv1)
	testsets.append(testsetv2)
	testsets.append(testsetv3)

	testloaders.append(testloaderv1)
	testloaders.append(testloaderv2)
	testloaders.append(testloaderv3)

	return trainset, trainloader, testsets, testloaders


def get_civilcomments(batch_size, data_dir, net, max_token_length):

	dataset = CivilCommentsDataset(download=True, root_dir=f"{data_dir}")
	
	trainset = dataset.get_subset('train', transform = initialize_bert_transform(net, max_token_length) )
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsets = []
	testloaders = []

	testsetv2 = dataset.get_subset('val', transform = initialize_bert_transform(net, max_token_length) )

	testsetv3 = dataset.get_subset('test', transform = initialize_bert_transform(net, max_token_length) )

	# testsets.append(testsetv1)
	testsets.append(testsetv2)
	testsets.append(testsetv3)

	def get_groups(testset): 
		group_testset = []
		for i in range(7): 
			groups = dataset._eval_groupers[i].metadata_to_group(testset.metadata_array)

			idx = np.where(groups==1)[0]
			group_testset.append(WILDSSubset(testset, idx, transform = None ))
			
			idx = np.where(groups==3)[0]
			group_testset.append(WILDSSubset(testset, idx, transform = None ))
		
		return group_testset

	testsetv3_groups = get_groups(testsetv3)

	testsets.extend(testsetv3_groups)

	for set in testsets:
		testloaders.append(torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=True,\
			 num_workers=2))

	return trainset, trainloader, testsets, testloaders


def get_imagenet(batch_size, data_dir, train=False, eval=True):


	imagenet_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur",\
				 "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate",\
				  "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate" ]
	severities = [1, 2, 3, 4 ,5]

	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	
	trainset = None 
	trainloader = None 

	if train: 
		trainset = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv1/train/", transform = transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsets = []
	testloaders = []

	if eval:
		testsetv1 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv1/val/", transform = transform)
		testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True,\
			num_workers=2)

		testsetv2 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-matched-frequency-format-val/", transform = transform)
		testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True,\
			num_workers=2)

		testsetv2_1 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-threshold0.7-format-val/", transform = transform)
		testloaderv2_1 = torch.utils.data.DataLoader(testsetv2_1, batch_size=batch_size, shuffle=True,\
			num_workers=2)

		testsetv2_2 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-top-images-format-val", transform = transform)
		testloaderv2_2 = torch.utils.data.DataLoader(testsetv2_2, batch_size=batch_size, shuffle=True,\
			num_workers=2)

		testsetv3 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet-sketch/", transform = transform)
		testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True,\
			num_workers=2)

		testsets.append(testsetv1)
		testsets.append(testsetv2)
		testsets.append(testsetv2_1)
		testsets.append(testsetv2_2)
		testsets.append(testsetv3)

		testloaders.append(testloaderv1)
		testloaders.append(testloaderv2)
		testloaders.append(testloaderv2_1)
		testloaders.append(testloaderv2_2)
		testloaders.append(testloaderv3)

		for data in imagenet_c:
			for severity in severities: 
				testset = torchvision.datasets.ImageFolder(root=f"{data_dir}/imagenet-c/" + data + "/" + str(severity),\
					transform=transform)
				testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True,\
					num_workers=2)

				testsets.append(testset)
				testloaders.append(testloader)	

	return trainset, trainloader, testsets, testloaders


def get_imagenet200(batch_size, data_dir, train=False, eval = True):

	imagenet_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur",\
				 "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate",\
				  "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate" ]
	severities = [1, 2, 3, 4 ,5]

	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		])
	
	trainset = None 
	trainloader = None 

	if train: 
		trainset = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv1/train/", transform = transform)
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,\
		 num_workers=2)

	testsets = []
	testloaders = []

	if eval:
		testsetv1 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv1/val/", transform = transform)
		testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True,\
			num_workers=2)

		testsetv2 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-matched-frequency-format-val/", transform = transform)
		testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True,\
			num_workers=2)

		testsetv2_1 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-threshold0.7-format-val/", transform = transform)
		testloaderv2_1 = torch.utils.data.DataLoader(testsetv2_1, batch_size=batch_size, shuffle=True,\
			num_workers=2)

		testsetv2_2 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-top-images-format-val", transform = transform)
		testloaderv2_2 = torch.utils.data.DataLoader(testsetv2_2, batch_size=batch_size, shuffle=True,\
			num_workers=2)


		testsetv3 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet-sketch/", transform = transform)
		testloaderv3 = torch.utils.data.DataLoader(testsetv3, batch_size=batch_size, shuffle=True,\
			num_workers=2)

		testsetv4 = torchvision.datasets.ImageFolder(f"{data_dir}/imagenet-r/", transform = transform)
		testloaderv4 = torch.utils.data.DataLoader(testsetv4, batch_size=batch_size, shuffle=True,\
			num_workers=2)

		testsets.append(testsetv1)
		testsets.append(testsetv2)
		testsets.append(testsetv2_1)
		testsets.append(testsetv2_2)
		testsets.append(testsetv3)
		testsets.append(testsetv4)

		testloaders.append(testloaderv1)
		testloaders.append(testloaderv2)
		testloaders.append(testloaderv2_1)
		testloaders.append(testloaderv2_2)
		testloaders.append(testloaderv3)
		testloaders.append(testloaderv4)

		for data in imagenet_c:
			for severity in severities: 
				testset = torchvision.datasets.ImageFolder(root=f"{data_dir}/imagenet-c/" + data + "/" + str(severity),\
					transform=transform)
				testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True,\
					num_workers=2)

				testsets.append(testset)
				testloaders.append(testloader)	

	return trainset, trainloader, testsets, testloaders

def get_imagenet_breeds(batch_size, data_dir, name = None):
	
	from robustness.tools.helpers import get_label_mapping
	from robustness.tools import folder
	from robustness.tools.breeds_helpers import make_living17, make_entity13, make_entity30, make_nonliving26


	if name == "living17": 
		ret = make_living17(f"{data_dir}/imagenet_class_hierarchy/", split="good")
	elif name == "entity13":
		ret = make_entity13(f"{data_dir}/imagenet_class_hierarchy/", split="good")
	elif name == "entity30":
		ret = make_entity30(f"{data_dir}/imagenet_class_hierarchy/", split="good")
	elif name == "nonliving26":
		ret = make_nonliving26(f"{data_dir}/imagenet_class_hierarchy/", split="good")

	keep_ids = np.array(ret[1]).reshape((-1))

	# merge_label_mapping = get_label_mapping('custom_imagenet', np.concatenate((ret[1][0], ret[1][1]), axis=1)) 
	source_label_mapping = get_label_mapping('custom_imagenet', ret[1][0]) 
	target_label_mapping = get_label_mapping('custom_imagenet', ret[1][1])

	transform = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize([0.4717, 0.4499, 0.3837], [0.2600, 0.2516, 0.2575])
		])
	
	trainset = None 
	trainloader = None 

	trainset = folder.ImageFolder(root=f"{data_dir}/imagenetv1/train/", transform = transform, label_mapping = source_label_mapping)
	testset = folder.ImageFolder(root=f"{data_dir}/imagenetv1/train/", transform = transform, label_mapping = target_label_mapping)

	imagenet_c = ["fog", "frost", "motion_blur", "brightness", "zoom_blur", "snow", "defocus_blur", "glass_blur",\
				 "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate",\
				  "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate" ]
	severities = [1, 2, 3, 4 ,5]


	idx = np.arange(len(trainset))
	np.random.seed(42)
	np.random.shuffle(idx)

	train_idx = idx[:len(idx)-10000]
	val_idx = idx[len(idx)-10000:]

	train_subset = torch.utils.data.Subset(trainset, train_idx)
	test_subset = torch.utils.data.Subset(trainset, val_idx)

	trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True,\
		num_workers=2)

	testsets = []
	testloaders = []

	testloaderv0_1 = torch.utils.data.DataLoader(test_subset, batch_size=batch_size, shuffle=True,\
		num_workers=2)
	
	testloaderv0_2 = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True,\
		num_workers=2)


	testsetv1 = folder.ImageFolder(f"{data_dir}/imagenetv1/val/", transform = transform, label_mapping = source_label_mapping)
	testloaderv1 = torch.utils.data.DataLoader(testsetv1, batch_size=batch_size, shuffle=True,\
		num_workers=2)

	testsetv2 = folder.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-matched-frequency-format-val/", transform = transform, label_mapping = source_label_mapping)
	testloaderv2 = torch.utils.data.DataLoader(testsetv2, batch_size=batch_size, shuffle=True,\
		num_workers=2)

	testsetv2_1 = folder.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-threshold0.7-format-val/", transform = transform, label_mapping = source_label_mapping)
	testloaderv2_1 = torch.utils.data.DataLoader(testsetv2_1, batch_size=batch_size, shuffle=True,\
		num_workers=2)

	testsetv2_2 = folder.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-top-images-format-val", transform = transform, label_mapping = source_label_mapping)
	testloaderv2_2 = torch.utils.data.DataLoader(testsetv2_2, batch_size=batch_size, shuffle=True,\
		num_workers=2)
	
	testsets.append(test_subset)
	testsets.append(testset)
	testsets.append(testsetv1)
	testsets.append(testsetv2)
	testsets.append(testsetv2_1)
	testsets.append(testsetv2_2)

	testloaders.append(testloaderv0_1)
	testloaders.append(testloaderv0_2)
	testloaders.append(testloaderv1)
	testloaders.append(testloaderv2)
	testloaders.append(testloaderv2_1)
	testloaders.append(testloaderv2_2)

	for data in imagenet_c:
		for severity in severities: 
			testset = folder.ImageFolder(root=f"{data_dir}/imagenet-c/" + data + "/" + str(severity),\
				transform=transform, label_mapping = source_label_mapping)
			testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True,\
				num_workers=2)

			testsets.append(testset)
			testloaders.append(testloader)	

	testsetv1_t = folder.ImageFolder(f"{data_dir}/imagenetv1/val/", transform = transform, label_mapping = target_label_mapping)
	testloaderv1_t = torch.utils.data.DataLoader(testsetv1_t, batch_size=batch_size, shuffle=True,\
		num_workers=2)

	testsetv2_t = folder.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-matched-frequency-format-val/", transform = transform, label_mapping = target_label_mapping)
	testloaderv2_t = torch.utils.data.DataLoader(testsetv2_t, batch_size=batch_size, shuffle=True,\
		num_workers=2)

	testsetv2_1_t = folder.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-threshold0.7-format-val/", transform = transform, label_mapping = target_label_mapping)
	testloaderv2_1_t = torch.utils.data.DataLoader(testsetv2_1_t, batch_size=batch_size, shuffle=True,\
		num_workers=2)

	testsetv2_2_t = folder.ImageFolder(f"{data_dir}/imagenetv2/imagenetv2-top-images-format-val", transform = transform, label_mapping = target_label_mapping)
	testloaderv2_2_t = torch.utils.data.DataLoader(testsetv2_2_t, batch_size=batch_size, shuffle=True,\
		num_workers=2)

	testsets.append(testsetv1_t)
	testsets.append(testsetv2_t)
	testsets.append(testsetv2_1_t)
	testsets.append(testsetv2_2_t)

	testloaders.append(testloaderv1_t)
	testloaders.append(testloaderv2_t)
	testloaders.append(testloaderv2_1_t)
	testloaders.append(testloaderv2_2_t)

	for data in imagenet_c:
		for severity in severities: 
			testset_t = folder.ImageFolder(root=f"{data_dir}/imagenet-c/" + data + "/" + str(severity),\
				transform=transform, label_mapping = target_label_mapping)
			testloader_t = torch.utils.data.DataLoader(testset_t, batch_size=batch_size, shuffle=True,\
				num_workers=2)

			testsets.append(testset_t)
			testloaders.append(testloader_t)
			
	return trainset, trainloader, testsets, testloaders




