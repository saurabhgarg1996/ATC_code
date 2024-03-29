import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import numpy as np
import random 
from absl import app, flags
import time 
import os 

from calibration import *
from model_helper import *
from data_helper import *
from ATC_helper import *
from predict_acc_helper import *

FLAGS = flags.FLAGS
flags.DEFINE_string('data', "CIFAR", 'Dataset.')
flags.DEFINE_string('data_dir', "./data", 'Dataset Dir.')
flags.DEFINE_string('net', "ResNet18", 'Network to train.')
flags.DEFINE_integer('bs', 200, 'Batch Size.')
flags.DEFINE_integer('numClasses', 10, 'Number of classes.')
flags.DEFINE_string('ckpt_dir', "models_checkpoint", 'Checkpoint directory.')
flags.DEFINE_integer('startEpoch', 0, 'Start Epoch.')
flags.DEFINE_integer('endEpoch', 341, 'End Epoch.')
flags.DEFINE_integer('gapEpoch', 50, 'Freq Epoch.')
flags.DEFINE_integer('max_token_length', 512, 'MAx token length used for BERT models.')
flags.DEFINE_integer('seed', 1111, 'Epochs to train.')
flags.DEFINE_boolean('pretrained', False, 'Whether we want a pretrained model.')


def main(_):
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	torch.manual_seed(int(FLAGS.seed))
	torch.cuda.manual_seed(int(FLAGS.seed))
	np.random.seed(int(FLAGS.seed))
	random.seed(int(FLAGS.seed))

	epochs = [i for i in range(FLAGS.startEpoch,FLAGS.endEpoch,FLAGS.gapEpoch)]

	# acc_logfile = dir_names[0] + "/acc.csv"

	if not os.path.exists(FLAGS.ckpt_dir):
		print("Model directory {} does not exist.".format(FLAGS.ckpt_dir))
		exit() 



	print('==> Preparing data..')
	_, _, _ , testloaders = get_data(FLAGS.data_dir, FLAGS.data, FLAGS.bs, FLAGS.net, train=False, max_token_length=FLAGS.max_token_length)

	## Model
	print('==> Building model..')
	net = get_net(net=FLAGS.net, data=FLAGS.data, num_classes=FLAGS.numClasses, pretrained=FLAGS.pretrained)
	net = net.to(device)

	if device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True
	
	print("==> Evaluating ...")
	## Training and Testing Model
	for epoch in epochs: 
		print(f"Epoch {epoch} ... ")

		net.load_state_dict(torch.load( FLAGS.ckpt_dir +  "/ckpt-" + str(epoch) + ".pth"))

		probsv1, labelsv1 = save_probs(net, testloaders[0], device)

		pred_idxv1 = np.argmax(probsv1, axis=-1)
		pred_probsv1 = np.max(probsv1, axis=-1)
		v1acc = np.mean(pred_idxv1 == labelsv1)*100.

		try:
			calibrator = TempScaling()
			calibrator.fit(inverse_softmax(probsv1), labelsv1)
		except: 
			class Calibration: pass 
			calibrator = Calibration()
			calibrator.calibrate = lambda x: x

		calib_probsv1 = softmax(inverse_softmax(probsv1))


		calib_pred_idxv1 = np.argmax(calib_probsv1, axis=-1)
		calib_pred_probsv1 = np.max(calib_probsv1, axis=-1)
		

		# entropy = get_entropy(probsv1)
		calib_entropy = get_entropy(calib_probsv1)

		# _, entropy_thres_balance = find_threshold_balance(entropy, pred_idxv1 == labelsv1 )
		_, calib_entropy_thres_balance = find_ATC_threshold(calib_entropy, calib_pred_idxv1 == labelsv1 )
		# _, thres_balance = find_threshold_balance(pred_probsv1, pred_idxv1 == labelsv1 )
		_, calib_thres_balance = find_ATC_threshold(calib_pred_probsv1, calib_pred_idxv1 == labelsv1 )


		for i, testloader in enumerate(testloaders[1:]): 
			probs_new, labels_new = save_probs(net, testloader, device) 

			pred_idx_new = np.argmax(probs_new, axis=-1)
			pred_probs_new = np.max(probs_new, axis=-1)

			calib_probs_new = softmax(inverse_softmax(probs_new))
			calib_pred_idx_new = np.argmax(calib_probs_new, axis=-1)
			calib_pred_probs_new = np.max(calib_probs_new, axis=-1)

			# import pdb; pdb.set_trace()
			entropy_new = get_entropy(probs_new)
			calib_entropy_new = get_entropy(calib_probs_new)

			# entropy_pred_balance = get_acc(entropy_thres_balance, entropy_new)
			# entropy_conf_balance = num_corr(pred_idx_new, entropy_new, entropy_thres_balance, labels_new)
		
			calib_entropy_pred_balance = get_ATC_acc(calib_entropy_thres_balance, calib_entropy_new)
			# calib_entropy_conf_balance = num_corr(calib_pred_idx_new, calib_entropy_new, calib_entropy_thres_balance, labels_new)

			# pred_balance = get_acc(thres_balance, pred_probs_new)
			# conf_balance = num_corr(pred_idx_new, pred_probs_new, thres_balance, labels_new)

			calib_pred_balance = get_ATC_acc(calib_thres_balance, calib_pred_probs_new)
			# calib_conf_balance = num_corr(calib_pred_idx_new, calib_pred_probs_new, calib_thres_balance, labels_new)

			test_acc =  np.mean(pred_idx_new == labels_new)*100.0
			# calib_test_acc =  np.mean(calib_pred_idx_new == labels_new)*100.0

			# avg_conf = np.mean(pred_probs_new)*100.0
			calib_avg_conf = np.mean(calib_pred_probs_new)*100.0

			# doc_feat = v1acc + get_doc(pred_probsv1, pred_probs_new)*100.0
			calib_doc_feat = v1acc + get_doc(calib_pred_probsv1, calib_pred_probs_new)*100.0

			# im_estimate = get_im_estimate(pred_probsv1, pred_probs_new, (pred_idxv1 == labelsv1)) 
			calib_im_estimtate = get_im_estimate(calib_pred_probsv1, calib_pred_probs_new, (calib_pred_idxv1 == labelsv1)) 

			# with open(logFile, "a") as f:
			# 	f.write(("{:.4f}, {:.4f}, {:.4f},{:.4f}, {:.4f},{:.4f},{:.4f},{:.4f},{:.4f}," + \
			# 	"{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}," + \
			# 	"{:.4f}\n").format(test_acc, calib_test_acc, pred_balance, conf_balance, 
			# 	entropy_pred_balance, entropy_conf_balance, calib_pred_balance, calib_conf_balance, \
			# 	calib_entropy_pred_balance, calib_entropy_conf_balance,\
			# 	avg_conf, calib_avg_conf, doc_feat, calib_doc_feat, im_estimate, calib_im_estimtate))

			print(f"Dataset {i}, Test accuracy {test_acc:.2f}, ATC (MC) {calib_pred_balance:.2f}, ATC (NE) {calib_entropy_pred_balance:.2f}")


if __name__ == '__main__':
	app.run(main)