import numpy as np
import math
import torch
import os
import json
import math
from numpy.linalg import inv
import torch.nn.functional as F



# D^ and diag matrix
layer_var_m = np.load(open('./layer_var.npy','rb'))  # e.g. (512,512,3,3) ; (1000,2048) (output_channels, input_channels, filter size, filter size)

layer_cov_mat = np.load(open('./layer_cov_mat.npy','rb')) # e.g. (k,512*512*3*3) , the transpose of cov_mat ; (k,2048000) ; (k,output_channels*input_channels*filter_size*filter_size)


def H(channel_id,layer_var_m,layer_cov_mat):

	var_m = layer_var_m[channel_id] # output_channel_id (1,2048)

	diag_m = torch.diag(torch.tensor(var_m.ravel())).numpy()  #(2048,2048)
	print('diag_m shape : ',diag_m.shape)

	diag_m_inv = torch.diag(torch.tensor(1/var_m)).numpy()

	# print('layer_cov_mat shape : ',layer_cov_mat.shape) #(20,2048000)

	cov_mat = layer_cov_mat.reshape((layer_cov_mat.shape[0],layer_var_m.shape[0],layer_var_m.shape[1]))[:,channel_id,:]

	# cov_mat = layer_cov_mat.reshape(layer_var_m.shape.insert(layer_cov_mat.shape[0],0))[:,channel_id,:]

	print('cov_mat shape : ',cov_mat.shape) #(20,2048)

	d_hat_m = np.transpose(cov_mat)

	k = d_hat_m.shape[1] # rank = 20

	kk = diag_m.shape[0] # dimensionality of the covariate_matrix

	# H(X) = (kk/2)*(1+ln(2*pi))+ln(det(covariate_matrix))/2

	# ln(det(cov_matrix)) = torch.logdet()

	logdet_diag = np.sum([np.log(e) for e in var_m.ravel()])

	sign, logdet_d_hat = np.linalg.slogdet(np.identity(k) + np.matmul(np.matmul(np.transpose(d_hat_m),diag_m_inv),d_hat_m)/(k-1))

	entropy = (kk/2)*(1+np.log(2*math.pi)) + logdet_diag/2 + sign*logdet_d_hat/2

	return entropy


def H_joint(channel_i, channel_j, layer_cov_mat, layer_var_m):

	var_m = np.concatenate((layer_var_m[channel_i].ravel(),layer_var_m[channel_j].ravel()),axis=0) # output_channel_id (1,2048)

	diag_m = torch.diag(torch.tensor(var_m)).numpy()  #(4096，4096)
	print('joint diag_m shape : ',diag_m.shape) 

	diag_m_inv = torch.diag(torch.tensor(1/var_m)).numpy()

	layer_cov_mat = layer_cov_mat.reshape((layer_cov_mat.shape[0],layer_var_m.shape[0],layer_var_m.shape[1]))

	print('layer_cov_mat shape : ',layer_cov_mat.shape) # （20,1000,2048)

	cov_mat = layer_cov_mat[:,[channel_i,channel_j],:] # (20,2,2048)

	cov_mat = cov_mat.reshape((cov_mat.shape[0],-1)) # (20,4096)

	print('joint cov_mat shape : ',cov_mat.shape)  

	d_hat_m = np.transpose(cov_mat)

	k = d_hat_m.shape[1] # rank = 20

	kk = diag_m.shape[0] # dimensionality of the covariate_matrix

	# H(X) = (kk/2)*(1+ln(2*pi))+ln(det(covariate_matrix))/2

	# ln(det(cov_matrix)) = torch.logdet()

	logdet_diag = np.sum([np.log(e) for e in var_m.ravel()])

	sign, logdet_d_hat = np.linalg.slogdet(np.identity(k) + np.matmul(np.matmul(np.transpose(d_hat_m),diag_m_inv),d_hat_m)/(k-1))

	entropy = (kk/2)*(1+np.log(2*math.pi)) + logdet_diag/2 + sign*logdet_d_hat/2

	return entropy


mi_matrix = np.zeros((1000,1000))

mi_matrix_list = []

for class_i in range(1000):

	for class_j in range(class_i,1000):

		print('class_i : ',class_i)

		mi = H(class_i,layer_var_m,layer_cov_mat) + H(class_j,layer_var_m,layer_cov_mat) - H_joint(class_i,class_j,layer_cov_mat,layer_var_m)

		print('mi : ',mi)

		mi_matrix[class_i,class_j] = mi

		mi_matrix[class_j,class_i] = mi

		mi_matrix_list.append({'class_i': class_i, 'class_j':class_j, 'data': mi})
		mi_matrix_list.append({'class_i': class_j, 'class_j':class_i, 'data': mi})


np.save(open('./mi_matrix.npy','wb'),mi_matrix)

json.dump(mi_matrix_list,open('./mi_matrix_list.json','w'))





















