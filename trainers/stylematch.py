import contextlib
import random
import os
import time
import datetime
import numpy as np
from math import ceil

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU, SimpleNet
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import build_transform
from dassl.utils import count_num_param

from .adain.adain import AdaIN

import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm
@contextlib.contextmanager
def freeze_models_params(models):
	try:
		for model in models:
			for param in model.parameters():
				param.requires_grad_(False)
		yield
	finally:
		for model in models:
			for param in model.parameters():
				param.requires_grad_(True)


class StochasticClassifier(nn.Module):
	def __init__(self, num_features, num_classes, temp=0.05):
		super().__init__()
		self.mu = nn.Parameter(0.01 * torch.randn(num_classes, num_features))
		self.sigma = nn.Parameter(torch.zeros(num_classes, num_features))
		self.temp = temp

	def forward(self, x, stochastic=True):
		mu = self.mu
		sigma = self.sigma

		if stochastic:
			sigma = F.softplus(sigma - 4)  # when sigma=0, softplus(sigma-4)=0.0181
			weight = sigma * torch.randn_like(mu) + mu
		else:
			weight = mu

		weight = F.normalize(weight, p=2, dim=1)
		x = F.normalize(x, p=2, dim=1)

		score = F.linear(x, weight)
		score = score / self.temp

		return score


class NormalClassifier(nn.Module):
	def __init__(self, num_features, num_classes):
		super().__init__()
		self.linear = nn.Linear(num_features, num_classes)

	def forward(self, x, stochastic=True):
		return self.linear(x)

# class Discriminator(nn.Module):
# 	def __init__(self):
# 		super(Discriminator, self).__init__()
# 		self.main = nn.Sequential(
# 			nn.Linear(512, 512),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Dropout(0.3),
# 			nn.Linear(512, 256),
# 			nn.LeakyReLU(0.2, inplace=True),
# 			nn.Dropout(0.3),
# 			nn.Linear(256, 1),
# 			nn.Sigmoid()
# 		)

# 	def forward(self, x):
# 		return self.main(x)



@TRAINER_REGISTRY.register()
class StyleMatch(TrainerXU):
	"""StyleMatch for semi-supervised domain generalization.

	Reference:
		Zhou et al. Semi-Supervised Domain Generalization with
		Stochastic StyleMatch. ArXiv preprint, 2021.
	"""

	def __init__(self, cfg):
		super().__init__(cfg)
		# Confidence threshold
		self.conf_thre = cfg.TRAINER.STYLEMATCH.CONF_THRE
		self.conf_thre = 0.1

		# Inference mode: 1) deterministic 2) ensemble
		self.inference_mode = cfg.TRAINER.STYLEMATCH.INFERENCE_MODE
		self.n_ensemble = cfg.TRAINER.STYLEMATCH.N_ENSEMBLE
		if self.inference_mode == "ensemble":
			print(f"Apply ensemble (n={self.n_ensemble}) at test time")

		norm_mean = None
		norm_std = None

		if "normalize" in cfg.INPUT.TRANSFORMS:
			norm_mean = cfg.INPUT.PIXEL_MEAN
			norm_std = cfg.INPUT.PIXEL_STD

		self.adain = AdaIN(
			cfg.TRAINER.STYLEMATCH.ADAIN_DECODER,
			cfg.TRAINER.STYLEMATCH.ADAIN_VGG,
			self.device,
			norm_mean=norm_mean,
			norm_std=norm_std,
		)

		self.apply_aug = cfg.TRAINER.STYLEMATCH.APPLY_AUG
		self.apply_sty = cfg.TRAINER.STYLEMATCH.APPLY_STY

		self.save_sigma = cfg.TRAINER.STYLEMATCH.SAVE_SIGMA
		self.sigma_log = {"raw": [], "std": []}
		if self.save_sigma:
			assert cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic"

	def check_cfg(self, cfg):
		assert len(cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS) > 0
		assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
		assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

	def build_data_loader(self):
		cfg = self.cfg
		tfm_train = build_transform(cfg, is_train=True)
		custom_tfm_train = [tfm_train]
		choices = cfg.TRAINER.STYLEMATCH.STRONG_TRANSFORMS
		tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
		custom_tfm_train += [tfm_train_strong]
		dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
		self.train_loader_x = dm.train_loader_x
		self.train_loader_u = dm.train_loader_u
		self.val_loader = dm.val_loader
		self.test_loader = dm.test_loader
		self.num_classes = dm.num_classes
		self.num_source_domains = dm.num_source_domains
		self.lab2cname = dm.lab2cname

	def build_model(self):
		cfg = self.cfg

		print("Building G")
		self.G = SimpleNet(cfg, cfg.MODEL, 0)  # n_class=0: only produce features
		self.G.to(self.device)
		print("# params: {:,}".format(count_num_param(self.G)))
		self.optim_G = build_optimizer(self.G, cfg.OPTIM)
		self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
		self.register_model("G", self.G, self.optim_G, self.sched_G)

		# print("Building D")
		# self.D = Discriminator().to(self.device)  # n_class=0: only produce features
		# print("# params: {:,}".format(count_num_param(self.D)))
		# self.optim_D = torch.optim.Adam(self.D.parameters(), lr=0.003)
		# self.sched_D = build_lr_scheduler(self.optim_D, cfg.OPTIM)
		# self.register_model("G", self.G, self.optim_G, self.sched_G)

		print("Building C")
		if cfg.TRAINER.STYLEMATCH.CLASSIFIER == "stochastic":
			self.C = StochasticClassifier(self.G.fdim, self.num_classes)
		else:
			self.C = NormalClassifier(self.G.fdim, self.num_classes)
		self.C.to(self.device)
		print("# params: {:,}".format(count_num_param(self.C)))
		self.optim_C = build_optimizer(self.C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
		self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.STYLEMATCH.C_OPTIM)
		self.register_model("C", self.C, self.optim_C, self.sched_C)

	def assess_y_pred_quality(self, y_pred, y_true, mask):
		n_masked_correct = (y_pred.eq(y_true).float() * mask).sum()
		acc_thre = n_masked_correct / (mask.sum() + 1e-5)  # accuracy after threshold
		acc_raw = y_pred.eq(y_true).sum() / y_pred.numel()  # raw accuracy
		keep_rate = mask.sum() / mask.numel()
		output = {"acc_thre": acc_thre, "acc_raw": acc_raw, "keep_rate": keep_rate}
		return output

	def L_cid1 (self,feature_matrix,labels,temperature):
		# torch.manual_seed(0)
		feature_matrix = feature_matrix.to(self.device)
		labels = labels.to(self.device)
		feature_matrix = torch.nn.functional.normalize(feature_matrix, p=2, dim=1)
		dot_product_matrix = torch.matmul(feature_matrix, feature_matrix.T)/temperature 
		# Compute the exponential of the dot product matrix
		exp_dot_product_matrix = torch.exp(dot_product_matrix)
		# Initialize tensors to hold the sums for same-class and different-class features
		same_class_sum = torch.zeros(len(labels))
		different_class_sum = torch.zeros(len(labels))

		# Compute the same-class and different-class masks
		for i in range(len(labels)):
			(labels == labels[i])
			# same_class_mask = (labels == labels[i])  # Same class mask
			same_class_mask = (labels == labels[i]) & (torch.arange(len(labels), device=self.device) != i)  # Same class mask excluding itself
			different_class_mask = (labels != labels[i])  # Different class mask
			# Sum the exponentials for same-class and different-class pairs
			same_class_sum[i] = exp_dot_product_matrix[i, same_class_mask].sum()
			different_class_sum[i] = exp_dot_product_matrix[i, different_class_mask].sum()

		# Compute the ratio and the negative logarithm of the ratio
		epsilon = 0.000001
		ratios = (same_class_sum+epsilon) / (different_class_sum + epsilon)
		results = -torch.log(ratios).to(self.device)
		return results.sum() #original is sum 
	
	def L_cid2 (self,features,labels,temperature):
		epsilon = 1e-8
		final_result = 0 
		features = torch.nn.functional.normalize(features, p=2, dim=1)
		# print ("f=")
		# print (features)
		# if torch.all(features <0 ):
		# 	print ("manfi")
		# 	exit()
		# Iterate over each item
		for i, class_label in enumerate(labels):
			# Get indices of all features belonging to the current class
			class_indices = torch.where(labels == class_label)[0]
			
			# If there are no or only one feature for the class, continue
			if len(class_indices) <= 1:
				continue
			
			# Gather the features for the current class
			class_features = features[class_indices]
			
			# Initialize sum for the current class
			sum_exp_same_class=0
			# Compute exp(matmul) for all pairs of features within the same class
			for j in range(len(class_indices)):
				if i!=j:
					matmul_result = torch.matmul(features[i].T, class_features[j])/temperature 
					# dot_product_matrix = torch.matmul(feature_matrix, feature_matrix.T)
					sum_exp_same_class += torch.exp(matmul_result)

			sum_all_others_exp = 0
			for j in range(len(labels)):
				if j!=i:
					matmul_result = torch.matmul(features[i].T, features[j])/temperature
					sum_all_others_exp += torch.exp(matmul_result)
			# print (sum_exp_same_class,sum_all_others_exp)
			ratios = (sum_exp_same_class+epsilon) / (sum_all_others_exp + epsilon)
			# print (ratios)
			result = -torch.log(ratios).to(self.device)
			
			final_result += result
			# exit()
		
		return final_result
		# return sum_exp_same_class, sum_exp_diff_class
	def L_cid(self,features, labels, temperature):
		epsilon = 1e-8
		# Normalize the features
		features = torch.nn.functional.normalize(features, p=2, dim=1)
		
		# Compute the dot product matrix (NxN)
		dot_product_matrix = torch.matmul(features, features.T) / temperature
		
		# Compute the exponentiated dot product matrix (NxN)
		exp_dot_product_matrix = torch.exp(dot_product_matrix)
		
		# Create a mask to zero out diagonal elements
		mask = torch.eye(features.size(0), device=features.device, dtype=torch.bool)
		
		# Calculate sum of exponentials for the same class
		same_class_mask = labels.unsqueeze(1) == labels.unsqueeze(0)
		same_class_mask = same_class_mask & ~mask  # Avoid self-comparison
		sum_exp_same_class = torch.sum(exp_dot_product_matrix * same_class_mask, dim=1) + epsilon
		
		# Calculate sum of all exponentials excluding the diagonal
		sum_all_others_exp = torch.sum(exp_dot_product_matrix * ~mask, dim=1) + epsilon
		
		# Compute the ratio and the loss
		ratios = sum_exp_same_class / sum_all_others_exp
		loss = -torch.log(ratios)
		
		# Return the mean loss
		return loss.mean()
	
	def decorre(self,matrix):
		# Step 1: Standardize the matrix
		mean = torch.mean(matrix, dim=0, keepdim=True)
		std = torch.std(matrix, dim=0, keepdim=True)
		epsilon = 0.00001
		standardized_matrix = (matrix - mean) / (std + epsilon)

		# Step 2: Compute the Gram matrix (Z^T * Z)
		gram_matrix = torch.matmul(standardized_matrix.t(), standardized_matrix)

		# Step 3: Subtract the diagonal elements to ignore i == j cases
		gram_matrix = gram_matrix - torch.diag(torch.diag(gram_matrix))

		# Step 4: Square the elements and sum them
		result = torch.sum(torch.square(gram_matrix))
	
		return result

		
	def uniform(self,matrix):
		# Step 1: Max normalize along each dimension (column-wise)
		# Compute the max value along each column
		max_values = torch.max(matrix, dim=0, keepdim=True)[0]

		# Avoid division by zero by replacing zeros with ones (if any)
		# max_values[max_values == 0] = 1

		# Normalize the feature matrix
		epsilon = 0.00001
		normalized_matrix = (matrix+epsilon) / (max_values+epsilon)

		# Step 2: Calculate the variance of the max-normalized features along each dimension (column-wise)
		variance_along_columns = torch.var(normalized_matrix, dim=0, unbiased=False)
		return -variance_along_columns.mean()

	def contrastive_loss(self,features, labels, margin=0.0):
		# Normalize the features
		features = F.normalize(features, p=2, dim=1)

		# Compute the cosine similarity matrix
		similarity_matrix = torch.mm(features, features.t())

		# Get the label mask
		labels = labels.view(-1, 1)
		label_mask = torch.eq(labels, labels.t()).float()
		# Compute the contrastive loss
		positive_loss = (1 - similarity_matrix) * label_mask
		negative_loss = F.relu(similarity_matrix - margin) * (1 - label_mask)

		loss = positive_loss.mean() + negative_loss.mean() 
		
		return loss
	
	def cosine_laplacian (self,X):
		epsilon = 1e-3
		X_norm = torch.nn.functional.normalize(X, p=2, dim=1,eps=epsilon)

		# Compute cosine similarity matrix
		cosine_similarity_matrix = torch.mm(X_norm, X_norm.t())

		# Step 2: Convert the cosine similarity matrix to an adjacency matrix
		# Here we assume the graph adjacency matrix is simply the cosine similarity matrix
		adjacency_matrix = cosine_similarity_matrix

		# Step 3: Compute the degree matrix
		degree_matrix = torch.diag(adjacency_matrix.sum(dim=1))

		# Step 4: Compute the Laplacian matrix
		laplacian_matrix = degree_matrix - adjacency_matrix
		
		return laplacian_matrix
	
	def laplacian_matrix(self,adj_matrix):
		degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
		laplacian = degree_matrix - adj_matrix
		return laplacian
	
	
	def cosine_adj(self,X):
		
		X_norm = F.normalize(X, p=2, dim=1)
		# Compute cosine similarity matrix
		cosine_similarity_matrix = torch.mm(X_norm, X_norm.t())
		return cosine_similarity_matrix
	
	def cosine_similarity_matrix_2(self,matrix1, matrix2, sigma=1.0):
		
		
		X_norm1 = F.normalize(matrix1, p=2, dim=1)
		X_norm2 = F.normalize(matrix2, p=2, dim=1)

		# Compute cosine similarity matrix
		cosine_similarity_matrix = torch.mm(X_norm1, X_norm2.t())
		return cosine_similarity_matrix
		
		
	
	def manifold_distance(self,adj_matrix1, adj_matrix2, Lst):
		laplacian_s = self.laplacian_matrix(adj_matrix1)
		laplacian_t = self.laplacian_matrix(adj_matrix2)
		laplacian_s_t = self.laplacian_matrix(Lst)
		epsilon = 1e-8
		laplacian_s = laplacian_s + (epsilon * torch.eye(laplacian_s.size(0))).to(self.device)
		laplacian_t = laplacian_t + epsilon * torch.eye(laplacian_t.size(0)).to(self.device)
		laplacian_s_t = laplacian_s_t + epsilon * torch.eye(laplacian_s_t.size(0)).to(self.device)
		
		eigenvalues_s, eigenvectors_s = torch.symeig(laplacian_s, eigenvectors=True)
		eigenvalues_t, eigenvectors_t = torch.symeig(laplacian_t, eigenvectors=True)
		diagonal_matrix_eigenvalues_s = torch.diag(eigenvalues_s)
		diagonal_matrix_eigenvalues_t = torch.diag(eigenvalues_t)

		##############
		# Check if the matrix is singular
		if any(diagonal_matrix_eigenvalues_t.diag() == 0):
			# Perturb the diagonal elements with a small positive constant
			
			diagonal_matrix_eigenvalues_t = diagonal_matrix_eigenvalues_t + (torch.eye(diagonal_matrix_eigenvalues_t.size(0)) * epsilon).to(self.device)

			# Now, the matrix should be invertible
			inverse_matrix_t = torch.inverse(diagonal_matrix_eigenvalues_t)
			
		else:
			# The matrix is already invertible
			inverse_matrix_t = torch.inverse(diagonal_matrix_eigenvalues_t)
		if any(diagonal_matrix_eigenvalues_s.diag() == 0):
			# Perturb the diagonal elements with a small positive constant
			epsilon = 1e-2
			diagonal_matrix_eigenvalues_s = diagonal_matrix_eigenvalues_s + (torch.eye(diagonal_matrix_eigenvalues_s.size(0)) * epsilon).to(self.device)

			# Now, the matrix should be invertible
			# inverse_matrix_s = torch.inverse(diagonal_matrix_eigenvalues_s)
			
		# else:
			# The matrix is already invertible
			# inverse_matrix_s = torch.inverse(diagonal_matrix_eigenvalues_s)
			
		##############
		approximated_eigenvector_S = laplacian_s_t @ eigenvectors_t @ inverse_matrix_t
		approximated_source_matrix = approximated_eigenvector_S @ diagonal_matrix_eigenvalues_t @ approximated_eigenvector_S.t() #eq10article
		difference = approximated_source_matrix - laplacian_s
		frobenius_norm = torch.norm(difference, p='fro')
		# normalized_frobenius_norm = frobenius_norm / torch.sqrt(torch.tensor(difference.size(0) * (difference.size(0))))	
		if torch.isnan(frobenius_norm):
			print ("nan umaddddd")
			print ("laplacian_s_t",laplacian_s_t)

			print ("approximated_source_matrix",approximated_source_matrix)
			print ("approximated_eigenvector_S",approximated_eigenvector_S)
			print ("diagonal_matrix_eigenvalues_t",diagonal_matrix_eigenvalues_t)
			print ("approximated_eigenvector_S",approximated_eigenvector_S)
			return torch.tensor(0.0, requires_grad=True)  # Return 0 if loss is NaN
		elif frobenius_norm == torch.inf:
			print ("inf umadddd")
			return torch.tensor(0.0, requires_grad=True)
		else:
			return frobenius_norm#.item()

	def denormalize(self,tensor, mean, std):
		mean = (torch.tensor(mean).reshape(1, -1, 1, 1)).to(self.device)
		std = (torch.tensor(std).reshape(1, -1, 1, 1)).to(self.device)
		return tensor * std + mean

	def show_images(self,images, n_cols=8):
		n_rows = len(images) // n_cols + int(len(images) % n_cols != 0)
		fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
		axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
		for i, image in enumerate(images):
			ax = axes[i]
			ax.imshow(image)
			ax.axis('off')
		for ax in axes[len(images):]:
			ax.axis('off')  # Hide any remaining empty subplots
		plt.show()

	def show_my_batch(self,XXX):
		mean = [0.5]  # Replace with your actual mean
		std = [0.5]   # Replace with your actual std
		# Denormalize the images
		for i in range(len(XXX)):
			print ("i=",i)
			images = self.denormalize(XXX[i], mean, std)

			# Convert the images to numpy arrays for visualization
			images = images.permute(0, 2, 3, 1).cpu().numpy()  # Change dimensions from (B, C, H, W) to (B, H, W, C)
			# Display the images
			self.show_images(images)

	def batch_normalize(self,features, eps=1e-5):
		# Calculate the mean and variance along the batch dimension
		mean = features.mean(dim=0, keepdim=True)
		variance = features.var(dim=0, unbiased=False, keepdim=True)
		
		# Normalize the features
		normalized_features = (features - mean) / torch.sqrt(variance + eps)
		
		return normalized_features

	

	def compute_discrimn_loss_empirical(self, W):
		"""Empirical Discriminative Loss."""
		self.gam1 = 1.0
		self.eps = 0.01
		p, m = W.shape
		I = torch.eye(p).cuda()
		scalar = p / (m * self.eps)
		logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
		print ("logdet=",logdet)
		return logdet / 2.

	def compute_compress_loss_empirical(self, W, Pi):
		"""Empirical Compressive Loss."""
		self.eps = 0.01
		p, m = W.shape
		k, _, _ = Pi.shape
		I = torch.eye(p).cuda()
		compress_loss = 0.
		for j in range(k):
			trPi = torch.trace(Pi[j]) + 1e-8
			scalar = p / (trPi * self.eps)
			# eps = 1e-6
			A = I + scalar * W.matmul(Pi[j]).matmul(W.T)
			A += self.eps * torch.eye(A.size(0)).to(self.device)
			A = A/2
			print ("max = ",torch.max(A))
			print ("det=",torch.det(A))
			log_det = torch.logdet(A)
			# log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
			print ("l=",log_det)
			compress_loss += log_det * trPi / m
		print ("comloss=",compress_loss)
		return compress_loss / 2.

	def one_hot(self,labels_int, n_classes):
		"""Turn labels into one hot vector of K classes. """
		labels_onehot = torch.zeros(size=(len(labels_int), n_classes)).float()
		for i, y in enumerate(labels_int):
			labels_onehot[i, y] = 1.0
		return labels_onehot

	def label_to_membership(self,targets, num_classes=None):
		
		targets = self.one_hot(targets, num_classes)
		num_samples, num_classes = targets.shape
		Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
		for j in range(len(targets)):
			k = np.argmax(targets[j])
			Pi[k, j, j] = 1.
		return Pi
	def k_nearest_affinity(self,matrix, k,num_feat_size):
		
		# Normalize the matrix for cosine similarity
		normalized_matrix = torch.nn.functional.normalize(matrix, dim=1)
		
		# Calculate cosine similarity (affinity matrix)
		affinity_matrix = torch.mm(normalized_matrix, normalized_matrix.T)
		#####################################################################
		#Calculate euclidean similarity 
		# sigma = 0.1
		# dist_matrix = torch.cdist(normalized_matrix, normalized_matrix, p=2) ** 2  # Squared Euclidean distances
	
		# # Compute Gaussian similarity
		# affinity_matrix = torch.exp(-dist_matrix / (2 * sigma ** 2))
		#####################################################################
		
		# Set negative values to 0 (retain only positive similarities)
		affinity_matrix = torch.where(affinity_matrix > 0, affinity_matrix, torch.zeros_like(affinity_matrix))
		# affinity_matrix.fill_diagonal_(0)
		# Retain only the k-nearest neighbors in each row
		#affinity_matrix[:,:num_feat_size]
		# print (affinity_matrix)
		if k != None:
			_, indices = torch.topk(affinity_matrix, k=k+1, dim=1)  # k+1 to include self-similarity #faghat unaee ke male labeled hastan tasirgozar bashan
			mask = torch.zeros_like(affinity_matrix)
			for i, row in enumerate(indices):
				mask[i, row] = 1
				mask[row,i] = 1 #jadid ezafe kardam vase tagharon 
			
			# Apply mask to retain only k-nearest neighbors
			k_nearest_affinity_matrix = affinity_matrix * mask
			return k_nearest_affinity_matrix
		else :
			return affinity_matrix
			
	def compute_Z1(self,W, Y, alpha):
		# Validate inputs
		if W.shape[0] != W.shape[1]:
			raise ValueError("Matrix W must be square.")
		if W.shape[0] != Y.shape[0]:
			raise ValueError("Matrix W and vector/matrix Y must have compatible dimensions.")
		
		# Step 1: Compute the degree matrix D (sum of rows of W)
		D = torch.diag(W.sum(dim=1))

		# Step 2: Compute D^(-1/2) and D^(1/2)
		D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag()))
		D_sqrt = torch.diag(torch.sqrt(D.diag()))

		# Step 3: Normalize using the formula D^(-1/2) W D^(1/2)
		W = D_inv_sqrt @ W @ D_sqrt

		# Identity matrix of the same size as W
		I = torch.eye(W.shape[0], device=W.device)
		
		# Compute (I - alpha * W)
		A = I - alpha * W
		
		# Compute the inverse of A
		try:
			A_inv = torch.linalg.inv(A)
		except RuntimeError:
			raise ValueError("Matrix (I - alpha * W) is singular and cannot be inverted.")
		
		# Compute Z
		Z = torch.matmul(A_inv, Y)
		
		return Z
	def compute_Z2(self,W, Y,alpha_labeled=0.01,alpha_unlabeled=0.99):
		alpha = alpha_unlabeled
		# Validate inputs
		if W.shape[0] != W.shape[1]:
			raise ValueError("Matrix W must be square.")
		if W.shape[0] != Y.shape[0]:
			raise ValueError("Matrix W and vector/matrix Y must have compatible dimensions.")
		
		# Step 1: Compute the degree matrix D (sum of rows of W)
		D = torch.diag(W.sum(dim=1))

		# Step 2: Compute D^(-1/2) and D^(1/2)
		D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag()))
		D_sqrt = torch.diag(torch.sqrt(D.diag()))
		
		L = (D-W).float()
		# Step 3: Normalize using the formula D^(-1/2) W D^(1/2)
		# W = D_inv_sqrt @ W @ D_sqrt

		W = D_inv_sqrt @ L @ D_sqrt

		# Identity matrix of the same size as W
		I = torch.eye(W.shape[0], device=W.device)
		
		# Compute (I - alpha * W)
		# A = I - alpha * W

		A = I + alpha * W
		
		# Compute the inverse of A
		try:
			A_inv = torch.linalg.inv(A)
		except RuntimeError:
			raise ValueError("Matrix (I - alpha * W) is singular and cannot be inverted.")
		
		# Compute Z
		
		Z = torch.matmul(A_inv, Y)
		
		return Z

	def compute_Z(self,W, Y, alpha_labeled,alpha_unlabeled):
		# Validate inputs
		if W.shape[0] != W.shape[1]:
			raise ValueError("Matrix W must be square.")
		if W.shape[0] != Y.shape[0]:
			raise ValueError("Matrix W and vector/matrix Y must have compatible dimensions.")
		
		# Step 1: Compute the degree matrix D (sum of rows of W)
		D = torch.diag(W.sum(dim=1))

		# Step 2: Compute D^(-1/2) and D^(1/2)
		# D_inv_sqrt = torch.diag(1.0 / D.diag())
		D_inv = torch.inverse(D)
		P = D_inv @ W
		alpha_labeled = 0.1
		alpha_unlabeled = 0.9 #0.9 bud
		I_alpha = torch.eye(W.shape[0], device=W.device) * alpha_labeled 
		# Identify the indices of the rows where the last column of Y is 1
		last_col_indices = (Y[:, -1] == 1).nonzero(as_tuple=True)[0]
		
		# Set the corresponding diagonal elements to zero
		I_alpha[last_col_indices, last_col_indices] = alpha_unlabeled
		
		I = torch.eye(W.shape[0], device=W.device) 

		I_beta = I-I_alpha
		inv_temp = torch.inverse(I-I_alpha@P)

		FF = inv_temp@I_beta@Y
		
		return FF

	def conjugate_gradient(self,W, Z, Y, tol=1e-5, max_iter=100):
		alpha = 0.99
		I = torch.eye(W.shape[0], device=W.device)
		
		# Compute (I - alpha * W)
		W = I - alpha * W
		n, m = Y.shape
		Z_est = Z.clone()  # Initialize Z with the same shape as the initial guess

		for j in range(m):  # Solve for each column independently
			# Extract column j
			y_col = Y[:, j]
			z_col = Z[:, j]
			
			# Initialize for conjugate gradient
			r = y_col - W @ z_col  # Residual
			p = r.clone()          # Search direction
			rs_old = torch.sum(r * r)  # Frobenius norm squared
			
			for i in range(max_iter):
				Ap = W @ p
				alpha = rs_old / torch.sum(p * Ap)  # Adjust for Frobenius norm
				z_col = z_col + alpha * p
				r = r - alpha * Ap
				rs_new = torch.sum(r * r)  # Frobenius norm squared
				
				if torch.sqrt(rs_new) < tol:
					break
				
				p = r + (rs_new / rs_old) * p
				rs_old = rs_new
			
			# Update the corresponding column in Z
			Z_est[:, j] = z_col

		return Z_est

	def entropy(self,matrix):
		
		# Check for any invalid values in the matrix
		if torch.any(torch.isnan(matrix)) or torch.any(torch.isinf(matrix)):
			raise ValueError("Input matrix contains NaN or Inf values!")
		matrix = torch.clamp(matrix, min=0.0)
		# Normalize rows to sum to 1 (handle all-zero rows by assigning uniform probabilities)
		row_sums = matrix.sum(dim=1, keepdim=True)
		matrix = torch.where(row_sums > 0, matrix / row_sums, torch.full_like(matrix, 1.0 / matrix.size(1)))
		
		# # Avoid exact zeros in the matrix for log calculation
		matrix = torch.clamp(matrix, min=1e-10)

		# Compute the entropy for each row
		entropy = -torch.sum(matrix * torch.log(matrix), dim=1)

		return entropy


	def masked_cross_entropy(self,predictions, ground_truth, mask1, mask2):
		
		# Compute cross-entropy loss for each sample
		cross_entropy_loss = F.cross_entropy(predictions, ground_truth, reduction='none')  # Shape: (N,)

		# Apply masks
		weighted_loss = cross_entropy_loss * mask2 #* mask1  # Shape: (N,)

		# Compute mean of the weighted loss
		mean_loss = weighted_loss.mean()

		return mean_loss
	
	def normalization (self,matrix):
		matrix = torch.clamp(matrix, min=0.0)
		# Normalize rows to sum to 1 (handle all-zero rows by assigning uniform probabilities)
		row_sums = matrix.sum(dim=1, keepdim=True)
		matrix = torch.where(row_sums > 0, matrix / row_sums, torch.full_like(matrix, 1.0 / matrix.size(1)))
		return matrix
	
	def differentiable_one_hot(self,tensor):
		# Apply softmax to get normalized probabilities
		soft_probs = torch.softmax(tensor, dim=-1)
		
		# Create a one-hot-like tensor using the straight-through estimator
		hard_probs = torch.zeros_like(soft_probs)
		hard_probs.scatter_(-1, soft_probs.argmax(dim=-1, keepdim=True), 1.0)
		
		# Combine hard_probs for forward pass and soft_probs for backward pass
		return (hard_probs - soft_probs) + soft_probs  #.detach()
		
	def forward_backward(self, batch_x, batch_u):
		parsed_batch = self.parse_batch_train(batch_x, batch_u)
		
		# 0 dog 1 elefent 2 girrafe 3 guitar 4 horse 5 house 6 person

		x0 = parsed_batch["x0"]
		x = parsed_batch["x"]
		x_aug = parsed_batch["x_aug"]
		y_x_true = parsed_batch["y_x_true"]

		u0 = parsed_batch["u0"]
		u = parsed_batch["u"]
		u_aug = parsed_batch["u_aug"]
		y_u_true = parsed_batch["y_u_true"]  # tensor

		K = self.num_source_domains
		# NOTE: If num_source_domains=1, we split a batch into two halves
		K = 2 if K == 1 else K

		####################
		# Generate pseudo labels & simillarity based labels
		####################
		with torch.no_grad():
			p_xu = []
			for k in range(K):
				x_k = x[k]
				u_k = u[k]
				xu_k = torch.cat([x_k, u_k], 0)
				z_xu_k = self.C(self.G(xu_k), stochastic=False)
				p_xu_k = F.softmax(z_xu_k, 1)
				p_xu.append(p_xu_k)
			p_xu = torch.cat(p_xu, 0)

			p_xu_maxval, y_xu_pred = p_xu.max(1)
			mask_xu = (p_xu_maxval >= self.conf_thre).float()

			y_xu_pred = y_xu_pred.chunk(K)
			mask_xu = mask_xu.chunk(K)

			# Calculate pseudo-label's accuracy
			y_u_pred = []
			mask_u = []
			for y_xu_k_pred, mask_xu_k in zip(y_xu_pred, mask_xu):
				y_u_pred.append(
					y_xu_k_pred.chunk(2)[1]
				)  # only take the 2nd half (unlabeled data)
				mask_u.append(mask_xu_k.chunk(2)[1])
			y_u_pred = torch.cat(y_u_pred, 0)
			mask_u = torch.cat(mask_u, 0)
			y_u_pred_stats = self.assess_y_pred_quality(y_u_pred, y_u_true, mask_u)
		####################
		# Supervised loss
		####################
		loss_x = 0
		for k in range(K):
			# mohtavaye batch ha shabihe ham nist 
			x_k = x[k]
			y_x_k_true = y_x_true[k]
			z_x_k = self.C(self.G(x_k), stochastic=True)
			loss_x += F.cross_entropy(z_x_k, y_x_k_true)
			
		####################
		# Unsupervised loss
		####################
		# self.G.train()
		# self.C.train()
		loss_u_aug = 0
		loss_u_feat_clas = 0
		l_cont_total = 0
		l_uniform_column = 0 
		entropy_total = 0
		manif_diff = 0 
		# mohtavaye har k fargh mikone masalan avvali ba zarrafe shoru mishe dovvomi ba guitar 
		for k in range(K):
			y_xu_k_pred = y_xu_pred[k]
			mask_xu_k = mask_xu[k]
			mask_u_k_pseudo = mask_xu_k[mask_xu_k.size(0)//2:]
			
			x_k = x[k]
			y_x_k_true = y_x_true[k]
			u_k = u[k] 
			xu_k = torch.cat([x_k, u_k], 0)
			
			# f_xu_k = self.G(xu_k)
			
			# Compute loss for strongly augmented data
			x_k_aug = x_aug[k]
			u_k_aug = u_aug[k]
			
			xu_k_aug = torch.cat([x_k_aug, u_k_aug], 0)
			
			f_xu_k_aug = self.G(xu_k_aug)
			f_u_k_aug = f_xu_k_aug[x_k.size(0):,:]
			f_u_k_aug_normalized = F.normalize(f_u_k_aug,p=2. ,dim=1)
			z_xu_k_aug = self.C(f_xu_k_aug, stochastic=True)
			loss = F.cross_entropy(z_xu_k_aug, y_xu_k_pred, reduction="none")
			loss = (loss * mask_xu_k).mean()
			loss_u_aug += loss

			# FBC Loss
			# feat = self.feat[k] #7*512 namayandehaye har class be tartib 
			
			x_k = x[k]
			u_k = u[k]
			y_pseudo_u = self.one_hot(y_xu_k_pred[x_k.size(0):],self.num_classes+1)
			y_pseudo_u_for_vis = y_xu_k_pred[x_k.size(0):]
			xu_k = torch.cat([x_k, u_k], 0)
			f_xu_k = self.G(xu_k)
			f_xu_k = F.normalize(f_xu_k,p=2. ,dim=1)

			f_x_k = f_xu_k[:x_k.size(0),:]
			f_u_k = f_xu_k[x_k.size(0):,:] 
			# f_u_k = F.normalize(self.G(u_k), p=2. ,dim=1)
			# similarity = torch.mm(f_xu_k, feat.t())
			knn = 5 #5 / None
			epsilon = 0.00001

			###################################################################
			other_domains = [i for i in range(K) if i != k]
			k2 = random.choice(other_domains)
			# x_k2 = x[k2]
			# xu_k2 = torch.cat([x_k, u_k], 0)
			# f_xu_k2 = F.normalize(self.G(xu_k), p=2. ,dim=1)
			# f_x_k2 = f_xu_k2[:x_k2.size(0),:]
			# y_x_k_true2 = y_x_true[k2]
			#############################loss1#######################################
			
			# feat1 = self.feat[k]
			feat1 = f_x_k
			feat_total = torch.cat((feat1,f_u_k))     #f_xu_k
			Y_total = torch.zeros(feat_total.size(0),self.num_classes+1)
			Y_total[:,-1] = 1 #vase ravesh jadide ke classe akharesh yani outlier 
			# labels_feat = torch.tensor(list(range(self.num_classes))) #y_x_k_true
			labels_feat = y_x_k_true
			onehot_label_feat = self.one_hot(labels_feat,self.num_classes+1)
			num_feat_size = feat1.size(0) #feat.size(0)+feat_k2.size(0)
			Y_total[:num_feat_size,:] = onehot_label_feat #torch.cat([onehot_label_feat,onehot_label_feat],0) #onehot_label_feat#
			
			A = self.k_nearest_affinity(feat_total,knn,num_feat_size)
			W = (A+A.t())/2.0
			# print (W)
			#Z = (I-alpha*W)^-1 Y
			Z_total = self.compute_Z2(W.to(self.device),Y_total.to(self.device),alpha_labeled=0.01,alpha_unlabeled=0.99)
			
			Z_u = Z_total[num_feat_size:,:]
			# Z_u = self.differentiable_one_hot(Z_u)
			probs = F.softmax(Z_u, dim=1)
	
			# Compute entropy: -sum(p_i * log(p_i))
			# entropy_1 = torch.mean(-torch.sum(probs * torch.log(probs + 1e-10), dim=-1))
			
			p_u_maxval, y_u_pred = Z_u.max(1)
			mask_u1 = (y_u_pred != self.num_classes).float()

			# row_equal = torch.mm(Z_u.to(self.device) , y_pseudo_u.to(self.device).t())
			# row_equal = row_equal.mean(dim=1)
			# # Convert to 0 (if equal) or 1 (if not equal)
			# loss1 = 1 - row_equal.float()
			
			# loss1 = F.mse_loss(Z_u.to(self.device), y_pseudo_u.to(self.device), reduction="none")
			# loss1 = loss1.mean(dim=1) #vase mse 
			loss1 = F.cross_entropy(Z_u.to(self.device), y_pseudo_u.to(self.device), reduction="none")

			final_mask = mask_u_k_pseudo #* mask_u1
			loss1_temp = loss1
			loss1 = (loss1 *final_mask ).mean()
			# loss1 = (loss1 *final_mask ).sum()
			# loss1 = (loss1+epsilon)/(final_mask.sum()+epsilon)
			# print ('##############loss1#####################')
			# print (y_u_pred)
			# print (y_pseudo_u_for_vis)
			
			# # print (mask_u_k_pseudo*mask_u1)
			# # print (loss1)
			# print (loss1_temp)
			# print ('###################################')
			############################loss2###########################################################
			
			# feat2 = self.feat[k2]
			feat2 = self.G(x[k2])
			feat_total = torch.cat((feat2,f_u_k))     #f_xu_k
			Y_total = torch.zeros(feat_total.size(0),self.num_classes+1)
			Y_total[:,-1] = 1 #vase ravesh jadide ke classe akharesh yani outlier 
			# labels_feat = torch.tensor(list(range(self.num_classes))) #y_x_k_true
			labels_feat = y_x_k_true
			onehot_label_feat = self.one_hot(labels_feat,self.num_classes+1)
			num_feat_size = feat2.size(0) #feat.size(0)+feat_k2.size(0)
			Y_total[:num_feat_size,:] = onehot_label_feat #torch.cat([onehot_label_feat,onehot_label_feat],0) #onehot_label_feat#
			
			A = self.k_nearest_affinity(feat_total,knn,num_feat_size)
			
			W = (A+A.t())/2.0
			#Z = (I-alpha*W)^-1 Y
			Z_total = self.compute_Z2(W.to(self.device),Y_total.to(self.device),alpha_labeled=0.2,alpha_unlabeled=0.8)
			
			Z_u = Z_total[num_feat_size:,:]
			# Z_u = self.differentiable_one_hot(Z_u)
			probs = F.softmax(Z_u, dim=1)
	
			# Compute entropy: -sum(p_i * log(p_i))
			# entropy_2 = torch.mean(-torch.sum(probs * torch.log(probs + 1e-10), dim=-1))
			
			p_u_maxval, y_u_pred = Z_u.max(1)
			mask_u2 = (y_u_pred != self.num_classes).float()
			
			# row_equal = torch.mm(Z_u.to(self.device) , y_pseudo_u.to(self.device).t())
			# row_equal = row_equal.mean(dim=1)
			# # Convert to 0 (if equal) or 1 (if not equal)
			# loss2 = 1 - row_equal.float()
			# loss2 = F.mse_loss(Z_u.to(self.device), y_pseudo_u.to(self.device), reduction="none")
			# loss2 = loss2.mean(dim=1) #vase mse 
			
			loss2 = F.cross_entropy(Z_u.to(self.device), y_pseudo_u.to(self.device), reduction="none")
			final_mask = mask_u_k_pseudo #* mask_u2
			loss2_temp = loss2
			loss2 = (loss2 * final_mask ).mean()
			# loss2 = (loss2 * final_mask ).sum()
			# loss2 = (loss2+epsilon)/(final_mask.sum()+epsilon)
			# print ('##############loss2#####################')
			# print (y_u_pred)
			# print (y_pseudo_u_for_vis)
			# # print (mask_u2)
			# # print (mask_u_k_pseudo)
			# # print (mask_u_k_pseudo*mask_u2)
			# # print (loss2)
			# print (loss2_temp)
			# print ('###################################################################################')
			loss_u_feat_clas += (loss1+loss2)/2.0
			###########################################################################################
			############### manifold diff #################
			graph_1 = self.cosine_adj(f_u_k)
			graph_2 = self.cosine_adj(f_u_k_aug_normalized)
			lst = self.cosine_similarity_matrix_2(f_u_k,f_u_k_aug_normalized)
			
			manif_diff += self.manifold_distance(graph_1,graph_2,lst)
			
			
		
		loss_summary = {}
		
		loss_all = 0
		loss_all += loss_x
		loss_summary["loss_x"] = loss_x.item()

		loss_all += loss_u_aug
		loss_summary["loss_u_aug"] = loss_u_aug.item()
		
		# print (loss_u_feat_clas)
		loss_all += 0.5*loss_u_feat_clas
		loss_summary["loss_u_FBC"] = loss_u_feat_clas.item()
		
		loss_all += 0.25*manif_diff
		loss_summary["manif_diff"] = manif_diff.item()

		self.model_backward_and_update(loss_all)

		
		loss_summary["y_u_pred_acc_thre"] = y_u_pred_stats["acc_thre"]
		loss_summary["y_u_pred_acc_raw"] = y_u_pred_stats["acc_raw"]
		loss_summary["y_u_pred_keep_rate"] = y_u_pred_stats["keep_rate"]


		if (self.batch_idx + 1) == self.num_batches:
			self.update_lr()
		return loss_summary

	def before_epoch(self):
		train_loader_x_iter = iter(self.train_loader_x)
		total_x = []
		total_y = []
		total_d = []
		for self.batch_idx in range(len(self.train_loader_x)):
			batch_x = next(train_loader_x_iter)

			input_x = batch_x["img0"]
			label_x = batch_x["label"]
			domain_x = batch_x["domain"]

			total_x.append(input_x)
			total_y.append(label_x)
			total_d.append(domain_x)

		x = torch.cat(total_x, dim=0)
		y = torch.cat(total_y, dim=0)
		d = torch.cat(total_d, dim=0)
		
		K = self.num_source_domains
		# NOTE: If num_source_domains=1, we split a batch into two halves
		K = 2 if K == 1 else K

		global_feat = []

		for i in range(K):
			idx = d == i
			imgs = x[idx]
			labels = y[idx]
			
			with torch.no_grad():
				z_imgs = self.G(imgs.to(self.device))
				z_imgs = F.normalize(z_imgs, p=2., dim=1)

			f = []
			for j in range(self.num_classes):
				idx = labels == j
				z = z_imgs[idx]
				# r = random.randint(0,z.size(0)-1)
				# f.append(z[r,:])
				f.append(z.mean(dim=0))
			feat = torch.stack(f)
			global_feat.append(feat)
		
		self.feat = torch.cat(global_feat, dim=0).chunk(K)
		# print ("featol=",len(self.feat),self.feat[0].shape) #mean of each class
		# exit()

		
	def parse_batch_train(self, batch_x, batch_u):
		x0 = batch_x["img0"]  # no augmentation
		x = batch_x["img"]  # weak augmentation
		x_aug = batch_x["img2"]  # strong augmentation
		y_x_true = batch_x["label"]
		
		x0 = x0.to(self.device)
		x = x.to(self.device)
		x_aug = x_aug.to(self.device)
		y_x_true = y_x_true.to(self.device)

		u0 = batch_u["img0"]
		u = batch_u["img"]
		u_aug = batch_u["img2"]
		y_u_true = batch_u["label"]  # for evaluating pseudo labeling's accuracy only

		u0 = u0.to(self.device)
		u = u.to(self.device)
		u_aug = u_aug.to(self.device)
		y_u_true = y_u_true.to(self.device)

		# Split data into K chunks
		K = self.num_source_domains
		# NOTE: If num_source_domains=1, we split a batch into two halves
		K = 2 if K == 1 else K
		x0 = x0.chunk(K)
		x = x.chunk(K)
		x_aug = x_aug.chunk(K)
		y_x_true = y_x_true.chunk(K)
		u0 = u0.chunk(K)
		u = u.chunk(K)
		u_aug = u_aug.chunk(K)

		batch = {
			# x
			"x0": x0,
			"x": x,
			"x_aug": x_aug,
			"y_x_true": y_x_true,
			# u
			"u0": u0,
			"u": u,
			"u_aug": u_aug,
			"y_u_true": y_u_true,  # kept intact
		}

		return batch

	def model_inference(self, input):
		features = self.G(input)

		if self.inference_mode == "deterministic":
			prediction = self.C(features, stochastic=False)

		elif self.inference_mode == "ensemble":
			prediction = 0
			for _ in range(self.n_ensemble):
				prediction += self.C(features, stochastic=True)
			prediction = prediction / self.n_ensemble

		else:
			raise NotImplementedError

		return prediction

	def after_train(self):
		print("Finish training")

		if self.epoch == self.max_epoch-1:
			torch.save(self.G, "G_feature_extractor_full_.pth")
			with open("/mnt/hard/home/atghaei/w/ssdg-benchmark/output/feat.pkl", "wb") as f:
				pickle.dump(self.feat, f)
			
		# Do testing
		else:
			if not self.cfg.TEST.NO_TEST:
				self.test()

			# Save model
			self.save_model(self.epoch, self.output_dir)
			
			# Show elapsed time
			elapsed = round(time.time() - self.time_start)
			elapsed = str(datetime.timedelta(seconds=elapsed))
			print("Elapsed: {}".format(elapsed))

			# Close writer
			self.close_writer()

			# Save sigma
			if self.save_sigma:
				sigma_raw = np.stack(self.sigma_log["raw"])
				np.save(os.path.join(self.output_dir, "sigma_raw.npy"), sigma_raw)

				sigma_std = np.stack(self.sigma_log["std"])
				np.save(os.path.join(self.output_dir, "sigma_std.npy"), sigma_std)





