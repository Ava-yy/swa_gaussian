import os
import numpy as np
import json
import argparse
import torchvision.models
from torchvision import datasets, models, transforms
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from PIL import Image



base_dir = '/home/zhaoy32/Desktop/swa_gaussian/food-101/'
image_swag_result_dir = os.path.join(base_dir,'image_swag_result/')
swag_all_images_result_dir = os.path.join(base_dir,'json_results2/')
# base_dir = base_dir if base_dir[-1] == '/' else base_dir+'/'

resnet50_model = torch.load(os.path.join(base_dir,"food101_finetune.pt"))

image_id_path_dict = json.load(open(os.path.join(base_dir,'image_id_path_dict.json'),'r'))


IMG_SIZE = (224, 224) 
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMG_MEAN, IMG_STD)
])

num_classes = 10

num_models = 100


def calculate_jacobian_avg(base_dir):

    jacobian_avg_dict = {}

    for image_id, value in image_id_path_dict.items():

        label = value['label']

        label_id = value['label_id']

        image_path = os.path.join(base_dir, 'images', image_id_path_dict[image_id]['image_path'].split('./eval_images/')[1])

        image = Image.open(image_path)

        image = test_transform(image)

        x = Variable(image,requires_grad=True)
        
        jacobian_avg = np.zeros((num_classes,x.shape[0],x.shape[1],x.shape[2]))
        
        x = x.cuda(non_blocking=True).unsqueeze(0)

        for sample_model_file in os.listdir(os.path.join(base_dir,'sample_models')):

            sample_id = sample_model_file.split('-')[1].split('.pt')[0]

            sample_model_dir = os.path.join(base_dir,'sample_models','swag_sample-'+sample_id+'.pt')

            checkpoint = torch.load(sample_model_dir)
            resnet50_model.load_state_dict(checkpoint["state_dict"])

            resnet50_model.eval()

            y = resnet50_model(x) # shape (1,10) tensor([[ -3.0459,  -9.2013,  -8.5121,   3.3400, -12.8477,  -6.3842,  -3.1321, -3.0611,  -0.2655,  -3.6536]], device='cuda:0', grad_fn=<AddmmBackward0>)

            for class_id in range(y.shape[1]):

                y_class = y[0][class_id]

                grad_x, = torch.autograd.grad(y_class, x, create_graph=True)

                jacobian = grad_x.reshape(x.shape) # (1,3,224,224) torch.Tensor

                jacobian = jacobian.cpu().detach().numpy()

                jacobian_avg[class_id] += jacobian[0]

        jacobian_avg /= num_models

        jacobian_avg_dict[image_id] = jacobian_avg

    json.dump(jacobian_avg_dict,open(os.path.join(base_dir,'jacobian_avg.json'),'w'))


if __name__ == '__main__':
    
    calculate_jacobian_avg(base_dir)
