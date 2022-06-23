import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

from models import *

import random
import numpy as np

def get_net(net, data, num_classes=10, pretrained=False): 
	change=True
	if net == "ResNet18": 
		net = ResNet18(num_classes=num_classes)
		change=False
	elif net == "LeNet": 
		net = LeNet()
	elif net == "ConvNet": 
		net = ConvNet()
	elif net == "FCN" and data=="MNIST": 
		net = nn.Sequential(nn.Flatten(),
				nn.Linear(28*28, 5000, bias=True),
				nn.ReLU(),
				nn.Linear(5000, 5000, bias=True),
				nn.ReLU(),
				nn.Linear(5000, 50, bias=True),
				nn.ReLU(),
				nn.Linear(50, num_classes, bias=True)
			)
	elif net == "DenseNet121" and data.startswith("CIFAR"): 
		net = DenseNet121(num_classes = num_classes)
		change=False
	elif net == "FCN": 
		net = nn.Sequential(nn.Flatten(),
				nn.Linear(32*32*3, 5000, bias=True),
				nn.ReLU(),
				nn.Linear(5000, 5000, bias=True),
				nn.ReLU(),
				nn.Linear(5000, 50, bias=True),
				nn.ReLU(),
				nn.Linear(50, num_classes, bias=True)
			)
	elif net == "AllConv": 
		net = AllConv(num_classes= num_classes)

	elif (net =="DenseNet121") or (net == "DenseNet" and data.startswith("CIFAR")) : 
		net = torchvision.models.densenet121(pretrained=pretrained)
		last_layer_name = 'classifier'
		# change = False
	elif net =="ResNet50": 
		net = torchvision.models.resnet50(pretrained=pretrained)
		last_layer_name = 'fc'
		# change = False

	elif net == "dummy": 
		net = nn.Sequential(nn.Flatten(),
				nn.Linear(96*96*3, num_classes, bias=True)
			)

	elif net == "distilbert-base-uncased": 
		net = initialize_bert_based_model(net, num_classes)
	
	if net in ('ResNet50', 'DenseNet', 'DenseNet121') and change:
		d_features = getattr(net, last_layer_name).in_features
		last_layer = nn.Linear(d_features, num_classes)
		net.d_out = num_classes
		setattr(net, last_layer_name, last_layer)

	return net	


def update_optimizer(epoch, opt, data, net, lr): 
	if data.startswith("CIFAR") or data.startswith("living17") or data.startswith("nonliving26"): 
		if epoch>=150: 
			for g in opt.param_groups:
				g['lr'] = 0.1*lr			
		if epoch>=250: 
			for g in opt.param_groups:
				g['lr'] = 0.01*lr

	if data.startswith("ImageNet_catdog") or data.startswith("entity13") or data.startswith("entity30"): 
		if epoch>=100: 
			for g in opt.param_groups:
				g['lr'] = 0.1*lr			
		if epoch>=200: 
			for g in opt.param_groups:
				g['lr'] = 0.01*lr


	# elif data=="Camelyon17": 
	elif data=="FMoW" or data=="ImageNet-200" or data=="Office31": 
		for g in opt.param_groups:
			g['lr'] = (0.96**(epoch))*lr

	
	elif data=="Amazon": 
		for g in opt.param_groups:
			g['lr'] = (3.0- epoch)/3.0 *lr
	
	elif data=="CivilComments": 
		for g in opt.param_groups:
			g['lr'] = (5.0- epoch)/5.0 *lr

	elif data=="RxRx1":
		if epoch <10: 
			for g in opt.param_groups:
				g['lr'] = (epoch+1)*lr / 10.0
		else: 
			for g in opt.param_groups:
				g['lr'] = max(0.0, 0.5*(1.0  + math.cos(math.pi *(epoch - 10.0/(80.0)))))*lr

	return opt


def train(net, loader, optimizer, criterion, device): 
	net.train()
	train_loss = 0
	correct = 0
	total = 0
	for batch_idx, batch_data  in enumerate(loader):
		inputs, targets = batch_data[0], batch_data[1]
		inputs, targets = inputs.to(device), targets.to(device)
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, targets)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		_, predicted = outputs.max(1)
		total += targets.size(0)

		correct_preds = predicted.eq(targets).cpu().numpy()
		correct += np.sum(correct_preds)
		
	return 100.*correct/total



def test(net, loader, criterion, device, return_probs=False): 
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	probs = None
	labels = None
	with torch.no_grad():
		for batch_idx, batch_data  in enumerate(loader):
			inputs, targets = batch_data[0], batch_data[1]
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)
			loss = criterion(outputs, targets)

			if return_probs: 
				if probs is None: 
					probs = F.softmax(outputs, dim = -1).detach().cpu().numpy()
					labels = targets.cpu().numpy()
				else: 
					probs = np.concatenate((probs,F.softmax(outputs, dim = -1).detach().cpu().numpy()), axis=0)
					labels = np.concatenate((labels,targets.cpu().numpy()), axis=0)

			test_loss += loss.item()
			_, predicted = outputs.max(1)
			total += targets.size(0)
			correct += predicted.eq(targets).sum().item()

	if return_probs:
		return 100.*correct/total, probs, labels
	else: 
		return 100.*correct/total


def evaluate_disagreement(net1, net2, loader, device, calibrator1=None, calibrator2=None): 
	net1.eval()
	net2.eval()

	correct = 0
	num_matches = 0
	total = 0
	with torch.no_grad():
		for batch_idx, batch_data  in enumerate(loader):
			inputs, targets = batch_data[0], batch_data[1]
			inputs, targets = inputs.to(device), targets.to(device)
			outputs1 = net1(inputs)
			outputs2 = net2(inputs)

			if calibrator1 is not None: 
				outputs1 = calibrator1(outputs1.detach().cpu().numpy())
			else: 
				outputs1 = outputs1.detach().cpu().numpy()

			if calibrator2 is not None: 
				outputs2 = calibrator2(outputs2.detach().cpu().numpy())
			else: 
				outputs2 = outputs2.detach().cpu().numpy()

			predicted1 = np.argmax(outputs1,axis=-1)
			predicted2 = np.argmax(outputs2,axis=-1)

			total += targets.size(0)
			correct += np.equal(predicted1, predicted2).sum().item()

			targets = targets.detach().cpu().numpy()

			num_matches += (np.equal(predicted1, targets) == np.equal(predicted1, predicted2)).sum().item()

	return 100.*correct/total, 100.*num_matches/total

def model_save(epoch, net, dir, freq=5): 

    if epoch%freq==0:
        torch.save(net.state_dict(), dir + "/ckpt-" + str(epoch) + ".pth")


def save_probs(net, loader, device): 
	net.eval()

	probs = None
	labels = None

	with torch.no_grad():
		for batch_idx, batch_data  in enumerate(loader):
			inputs, targets = batch_data[0], batch_data[1]
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = net(inputs)

			if probs is None: 
				probs = F.softmax(outputs, dim = -1).detach().cpu().numpy()
				labels = targets.cpu().numpy()
			else: 
				probs = np.concatenate((probs,F.softmax(outputs, dim = -1).detach().cpu().numpy()), axis=0)
				labels = np.concatenate((labels,targets.cpu().numpy()), axis=0)


	return probs, labels