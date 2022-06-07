import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import numpy as np
import random 
from absl import app, flags
import time 
import os 

from datasets import *
from helper import *
from getData import *
from predictAcc_helper import *

FLAGS = flags.FLAGS

flags.DEFINE_string('net', "ResNet18", 'Network to train.')
flags.DEFINE_string('data', "CIFAR", 'Data.')
flags.DEFINE_integer('numClasses', 10, 'Number of classes.')
flags.DEFINE_string('ckpt_dir1', "remote_checkpoint", 'Google cloud bucket dir.')
flags.DEFINE_string('ckpt_dir2', "remote_checkpoint", 'Google cloud bucket dir.')
flags.DEFINE_integer('seed', 1111, 'Epochs to train.')
flags.DEFINE_integer('startEpoch', 0, 'Start Epoch.')
flags.DEFINE_integer('endEpoch', 341, 'End Epoch.')
flags.DEFINE_integer('gapEpoch', 50, 'Freq Epoch.')
flags.DEFINE_integer('bs', 200, 'Batch Size.')
flags.DEFINE_integer('max_token_length', 512, 'MAx token length used for BERT models.')


def main(_):

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	torch.manual_seed(FLAGS.seed)
	torch.cuda.manual_seed(FLAGS.seed)
	np.random.seed(FLAGS.seed)
	random.seed(FLAGS.seed)

	epochs = [i for i in range(FLAGS.startEpoch,FLAGS.endEpoch,FLAGS.gapEpoch)]


	if not os.path.exists(FLAGS.ckpt_dir1):
		print(f"Model not found in {FLAGS.ckpt_dir1}")
		exit()

	if not os.path.exists(FLAGS.ckpt_dir2):
		print(f"Model not found in {FLAGS.ckpt_dir2}")
		exit()

	_, _, testsets, testloaders = get_data(FLAGS, train=False)

	criterion = nn.CrossEntropyLoss()

	## Model
	print('==> Building model..')
	net1 = get_net(FLAGS, num_classes=FLAGS.numClasses)
	net1 = net1.to(device)

	net2 = get_net(FLAGS, num_classes=FLAGS.numClasses)
	net2 = net2.to(device)
	
	if device == 'cuda':
		net1 = torch.nn.DataParallel(net1)
		net2 = torch.nn.DataParallel(net2)
		cudnn.benchmark = True

	for epoch in epochs: 
		net1.load_state_dict(torch.load(FLAGS.ckpt_dir1 +  "/ckpt-" + str(epoch) + ".pth"))
		net2.load_state_dict(torch.load(FLAGS.ckpt_dir2 +  "/ckpt-" + str(epoch) + ".pth"))

		true_acc_net1 = []
		true_acc_net2 = []

		for testloader in testloaders:
			test_acc1 = test(net1, testloader, criterion, device, return_probs=False)
			test_acc2 = test(net2, testloader, criterion, device, return_probs=False)
			true_acc_net1.append(test_acc1)
			true_acc_net2.append(test_acc2)

		pred_acc = []
		matches = []

		print(f"Evaluated Epoch {epoch}") 

		for i, testloader in enumerate(testloaders):
			v_pred_acc, v_matches = evaluate_disagreement(net1, net2, testloader, device)
			pred_acc.append(v_pred_acc)
			matches.append(v_matches)

			print(f"Dataset {i}, Test accuracy {true_acc_net1[i]} and GDE predicted accuracy {pred_acc}")

if __name__ == '__main__':
	app.run(main)