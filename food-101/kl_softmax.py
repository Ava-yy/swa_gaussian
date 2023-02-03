import os
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json

from scipy.stats import entropy

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from PIL import Image
from glob import glob
from tqdm import tqdm


BASE_DIR = '/home/zhaoy32/Desktop/swa_gaussian/food-101'

IMAGE_SWAG_RESULT_DIR = os.path.join(BASE_DIR,'image_swag_result/')

if not os.path.exists(IMAGE_SWAG_RESULT_DIR):

	os.mkdir(IMAGE_SWAG_RESULT_DIR)


def image_swag_result(base_dir,image_swag_result_dir):

# the softmax predictions from all swag sample model, and the finetuned model, for each image

	num_samples = 100 # 100 swag models

	num_classes = 10 # 10 classes for food-101

	swag_result_path = os.path.join(base_dir,'json_results2')

	finetune_test_result = json.load(open(os.path.join(base_dir,'food101_finetune_test_result.json'),'r'))

	temp = {} # a dict to store the softmax predictions from all swag models for each image

	for image_res_finetune in finetune_test_result:

		image_id = image_res_finetune['image_id']

		temp[image_id] =  np.zeros((num_samples,num_classes))


	for swag_sample_file in os.listdir(swag_result_path):

		swag_sample_res = json.load(open(os.path.join(swag_result_path,swag_sample_file),'r')) # a list of swag predictions for num of test images

		sample_id = swag_sample_res[0]['sample_id']

		for datum in swag_sample_res:

			image_id = datum['image_id']

			temp[image_id][sample_id] = np.array(datum['predict_softmax'])


	for image_res_finetune in finetune_test_result:
		
		image_item_res = {}
		image_idx = image_res_finetune['image_id']
		image_item_res['image_id'] = image_res_finetune['image_id']
		image_item_res['image_path'] = image_res_finetune['image_path']
		image_item_res['label_id'] = image_res_finetune['label_id']
		image_item_res['label'] = image_res_finetune['label']
		image_item_res['finetune_softmax'] = image_res_finetune['predict_softmax']
		image_item_res['swag_softmax'] = temp[image_idx].tolist()
		image_item_res['swag_pred'] = temp[image_idx].argmax(1).tolist()

		#print(image_item_res['swag_softmax'],image_item_res['swag_pred'])

		json.dump(image_item_res,open(image_swag_result_dir+'image_'+str(image_idx)+'.json','w'))



def kl_pair(base_dir,image_swag_result_dir,image_id, sample_i, sample_j):

	base = 2

	image_swag_data = json.load(open(image_swag_result_dir +'image_'+str(image_id)+'.json','r'))

	swag_softmax_i = np.array(image_swag_data['swag_softmax'][sample_i])

	swag_softmax_j = np.array(image_swag_data['swag_softmax'][sample_j])

	kl_div_pair = entropy(swag_softmax_i, swag_softmax_j, base=base)

	return kl_div_pair


def kl_image(base_dir,image_swag_result_dir):

	image_kl = {}

	for file in os.listdir(image_swag_result_dir):

		print(file)

		data = json.load(open(os.path.join(image_swag_result_dir,file),'r'))

		swag_sample_softmax = np.array(data['swag_softmax']) #(num of swag samples, num of classes)

		image_id = data['image_id']

		num_samples = swag_sample_softmax.shape[0]  # num of swag samples

		image_kl[image_id] = np.zeros((num_samples,num_samples))

		for sample_i in range(swag_sample_softmax.shape[0]):

			for sample_j in np.arange(sample_i, swag_sample_softmax.shape[0]):

				kl_pair_res = kl_pair(base_dir, image_swag_result_dir, image_id,sample_i,sample_j)

				image_kl[image_id][sample_i][sample_j] = kl_pair_res
				image_kl[image_id][sample_j][sample_i] = kl_pair_res

	for image_id,kl_matrix in image_kl.items():

		image_kl[image_id] = kl_matrix.tolist()

	json.dump(image_kl,open(os.path.join(base_dir,'image_swag_kl.json'),'w'))




if __name__ == '__main__':

	image_swag_result(BASE_DIR,IMAGE_SWAG_RESULT_DIR)

	kl_image(BASE_DIR,IMAGE_SWAG_RESULT_DIR)
















