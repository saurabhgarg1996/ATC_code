import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from transformers import AdamW


import numpy as np
import random 
from absl import app, flags
import time 
import os 

# from datasets import *
from helper import *
from getData import *

FLAGS = flags.FLAGS

flags.DEFINE_float('lr', 0.1, 'Learning rate.')
flags.DEFINE_float('wd', 5e-4, 'Weight Decay.')
flags.DEFINE_string('data_dir', './data/', 'Data director.')
flags.DEFINE_integer('bs', 200, 'Batch Size.')
flags.DEFINE_integer('numClasses', 10, 'Number of classes.')
flags.DEFINE_float('momentum', 0.9, 'Momentum.')
flags.DEFINE_string('optimizer', "SGD", 'Optimizer.')
flags.DEFINE_string('data', "CIFAR", 'Dataset.')
flags.DEFINE_string('net', "ResNet18", 'Network to train.')
flags.DEFINE_string('modelDir', "models_checkpoint", 'Logging for models.')
flags.DEFINE_integer('epochs', 350, 'Epochs to train.')
flags.DEFINE_integer('seed', 1111, 'Epochs to train.')
flags.DEFINE_integer('logFreq', 50, 'Frequency for logging.')
flags.DEFINE_integer('max_token_length', 512, 'MAx token length used for BERT models.')
flags.DEFINE_boolean('pretrained', False, 'Whether we want a pretrained model.')
flags.DEFINE_boolean('eval', False, 'Whether we want to evaluate model.')


def main(_):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	torch.manual_seed(FLAGS.seed)
	torch.cuda.manual_seed(FLAGS.seed)
	np.random.seed(FLAGS.seed)
	random.seed(FLAGS.seed)

	## Dirs for logging and model check-pointing
	# timestr = time.strftime("%Y%m%d-%H%M%S")
	dir_name = "{}_{}_{}_{}_{}_{}_{}".format(FLAGS.net, FLAGS.lr, FLAGS.wd, FLAGS.momentum, FLAGS.optimizer, FLAGS.epochs, FLAGS.seed)

	acc_logfile = dir_name + "/acc.csv"

	if not os.path.exists( FLAGS.modelDir + "/" + dir_name ):
		os.makedirs(FLAGS.modelDir +  "/" + dir_name )

	# Data
	# Get data from gloud to server in data folder 

	print('==> Preparing data..')
	trainset, trainloader, testsets, testloaders = get_data(FLAGS, eval = FLAGS.eval)

	## Model
	print('==> Building model..')
	net = get_net(FLAGS, num_classes=FLAGS.numClasses)
	net = net.to(device)

	if device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True
	
	criterion = nn.CrossEntropyLoss()
	
	if FLAGS.optimizer=="SGD":
		optimizer = optim.SGD(net.parameters(), lr=FLAGS.lr, momentum=FLAGS.momentum,\
			 weight_decay=FLAGS.wd)
			 
	elif FLAGS.optimizer=="Adam": 
		optimizer = optim.Adam(net.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.wd)

	elif FLAGS.optimizer=="AdamW": 
		no_decay = ['bias', 'LayerNorm.weight']
		params = [
            {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': FLAGS.wd},
            {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
		optimizer = AdamW(params, lr=FLAGS.lr)


	print("==> Training ...")
	## Training and Testing Model
	for epoch in range(FLAGS.epochs): 
		print(epoch)
		optimizer = update_optimizer(epoch, optimizer, FLAGS.data, FLAGS.net, FLAGS.lr)

		train_acc = train(net, trainloader, optimizer, criterion, device)

		test_accs = []

		if FLAGS.eval: 
			for testloader in testloaders: 
				test_acc = test(net, testloader, criterion, device)
				test_accs.append(test_acc)


		model_save(epoch, net, FLAGS.modelDir + "/" + dir_name, FLAGS.logFreq)


if __name__ == '__main__':
	app.run(main)