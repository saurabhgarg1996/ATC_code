import numpy as np
import random 
from absl import app, flags
import time 
import os 


def inverse_softmax(preds):
	preds[preds==0.0] = 1e-40
	preds = preds/np.sum(preds, axis=1)[:, None]
	return np.log(preds) - np.mean(np.log(preds),axis=1)[:,None]

def idx2onehot(a, k): 
	a = a.astype(int)
	b = np.zeros((a.size, k))
	b[np.arange(a.size),a] = 1
	
	return b

def find_threshold(probs, labels):
    sorted_idx = np.argsort(probs)
    
    sorted_probs = probs[sorted_idx]
    sorted_labels = labels[sorted_idx]
    
    fp = np.sum(labels==0)
#     print(fp)
    fn = 0.0
    
    min_fp_fn = fp + fn
    thres = 0.0
    for i in range(len(labels)): 
        if sorted_labels[i] == 0: 
            fp -= 1
        else: 
            fn += 1
        
#         print(fp, fn)
        if (fp + fn) < min_fp_fn: 
            min_fp_fn = fp + fn
            thres = sorted_probs[i]
            
    return min_fp_fn, thres

def find_threshold_balance(probs, labels): 
    sorted_idx = np.argsort(probs)
    
    sorted_probs = probs[sorted_idx]
    sorted_labels = labels[sorted_idx]
    
    fp = np.sum(labels==0)
#     print(fp)
    fn = 0.0
    
    min_fp_fn = np.abs(fp - fn)
    thres = 0.0
    min_fp = fp
    min_fn = fn
    for i in range(len(labels)): 
        if sorted_labels[i] == 0: 
            fp -= 1
        else: 
            fn += 1
        
#         print(fp, fn)
        if np.abs(fp - fn) < min_fp_fn: 
            min_fp = fp
            min_fn = fn
            min_fp_fn = np.abs(fp - fn)
            thres = sorted_probs[i]
    
    return min_fp_fn, thres

def find_threshold_normalize(probs, labels):
    sorted_idx = np.argsort(probs)
    
    sorted_probs = probs[sorted_idx]
    sorted_labels = labels[sorted_idx]
    
    num_corr = np.sum(labels==0)
    num_incor = np.sum(labels==1)
    fp = np.sum(labels==0)
#     print(fp)
    fn = 0.0
    
    min_fp_fn = fp* 1.0 /num_corr + fn*1.0 / num_incor
    thres = 0.0
    for i in range(len(labels)): 
        if sorted_labels[i] == 0: 
            fp -= 1
        else: 
            fn += 1
        
#         print(fp, fn)
        if (fp* 1.0 /num_corr + fn*1.0 / num_incor) < min_fp_fn: 
            min_fp_fn = fp* 1.0 /num_corr + fn*1.0 / num_incor
            thres = sorted_probs[i]
            
    return min_fp_fn, thres

def get_acc(thres, probs): 
    return np.mean(probs>=thres)*100.0

def num_corr(idx, probs, thres , true_idx): 

	corr = (probs>=thres)
	true_corr = (idx == true_idx) ==  corr
	return np.mean(true_corr)*100.0

def get_entropy(probs): 
	return np.sum( np.multiply(probs, np.log(probs + 1e-20))  , axis=1)

def get_avg_conf(probs): 
	return np.mean(probs)

def get_doc(probs1, probs2):
	return np.mean(probs2) - np.mean(probs1) 

class HistogramDensity: 
    def _histedges_equalN(self, x, nbin):
        npt = len(x)
        return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))
    
    def __init__(self, num_bins = 10, equal_mass=False):
        self.num_bins = num_bins 
        self.equal_mass = equal_mass
        
        
    def fit(self, vals): 
        
        if self.equal_mass:
            self.bins = self._histedges_equalN(vals, self.num_bins)
        else: 
            self.bins = np.linspace(0,1.0,self.num_bins+1)
    
        self.bins[0] = 0.0 
        self.bins[self.num_bins] = 1.0
        
        self.hist, bin_edges = np.histogram(vals, bins=self.bins, density=True)
    
    def density(self, x): 
        curr_bins = np.digitize(x, self.bins, right=True)
        
        curr_bins -= 1
        return self.hist[curr_bins]
    
def get_im_estimate(probs_source, probs_target, corr_source): 

	source_binning = HistogramDensity()
	source_binning.fit(probs_source)

	target_binning = HistogramDensity()
	target_binning.fit(probs_target)

	weights = target_binning.density(probs_source) / source_binning.density(probs_source)


	weights = weights/ np.mean(weights)
	# print(weights)
	return np.mean(weights*corr_source)*100.0

def get_gde(epoch, file_name, num_ve): 

	with open(file_name, "r") as f: 
		f.readline()
		indices_preds = np.arange(0,num_ve)*4 +3
		indices_conf = np.arange(0,num_ve)*4+4
		for line in f: 
			vals = line.rstrip().split(",")
			vals = [float(val) for val in vals]

			if vals[0] == epoch: 
				return np.take(vals, indices_preds), np.take(vals, indices_conf)

def softmax(preact, temp=1, biases=None):
    if (biases is None):
        biases = np.zeros(preact.shape[1])
    exponents = np.exp(preact/temp + biases[None,:])
    sum_exponents = np.sum(exponents, axis=1) 
    return exponents/sum_exponents[:,None]