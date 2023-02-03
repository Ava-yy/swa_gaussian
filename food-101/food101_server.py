import os
import flask
from flask import Flask, request
from flask_cors import CORS

import numpy as np
import sys
import json
from tqdm import tqdm

from PIL import Image
from torchvision import transforms
import torch
import torch.nn as nn


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



app = Flask(__name__)
CORS(app)


@app.route('/calculate_jacobian', methods=['GET','POST'])

def calculate_jacobian():

    client_data = flask.request.json

    image_id = client_data['image_id']

    sample_id = client_data['sample_id']

    class_id = client_data['class_id']

    sample_model_dir = os.path.join(base_dir,'sample_models','swag_sample-'+str(sample_id)+'.pt')

    checkpoint = torch.load(sample_model_dir)
    resnet50_model.load_state_dict(checkpoint["state_dict"])

    resnet50_model.eval()

    print('image_path',image_id_path_dict[image_id]['image_path'].split('./eval_images/')[1])

    image_path = os.path.join(base_dir, 'images', image_id_path_dict[image_id]['image_path'].split('./eval_images/')[1])

    image = Image.open(image_path)

    image = test_transform(image)

    x = Variable(image,requires_grad=True)

    x = x.cuda(non_blocking=True).unsqueeze(0)

    y = resnet50_model(x)

    y = y.reshape[0][class_id] #(1,10)

    grad_x, = torch.autograd.grad(y, x, create_graph=True)

    jacobian = grad_x.reshape(x.shape) #(1,3,224,224)

    print('jacobian_shape : ',jacobian.shape)

    return flask.jsonify({'jacobian':jacobian[0].tolist(),'sample_id':sample_id,'class_id':class_id,'image_id':image_id})
    


@app.route('/image_swag_kl', methods=['GET','POST'])

def get_image_swag_kl():

    # client_data = flask.request.json

    # image_id = client_data['image_id']

    data = json.load(open(base_dir+'image_swag_kl.json','r'))

    return flask.jsonify({'data':data})


@app.route('/image_swag_result', methods=['GET','POST'])

def get_image_swag_result():

    client_data = flask.request.json

    image_id = client_data['image_id']

    data = json.load(open(image_swag_result_dir+'image_'+str(image_id)+'.json','r'))

    return flask.jsonify({'image_id':image_id,'data':data})



@app.route('/swag_all_images_result', methods=['GET','POST'])

def get_swag_all_images_result():

    client_data = flask.request.json

    sample_id = client_data['sample_id']

    data = json.load(open(swag_all_images_result_dir+'result_dict_'+str(sample_id)+'.json','r'))

    return flask.jsonify({'sample_id':sample_id,'data':data})
    



@app.route('/calculate_jacobian_avg', methods=['GET','POST'])

def calculate_jacobian_avg():

    client_data = flask.request.json

    image_id = client_data['image_id']

    class_id = client_data['class_id']

    jacobian_avg = np.zeros((3,224,224))

    for sample_model_file in os.listdir(os.path.join(base_dir,'sample_models')):

        sample_model_dir = os.path.join(base_dir,'sample_models','swag_sample-'+str(sample_id)+'.pt')

        checkpoint = torch.load(os.path.join(sample_model_dir,sample_model_file))
        resnet50_model.load_state_dict(checkpoint["state_dict"])

    image_path = os.path.join(base_dir, image_id_path_dict[image_id]['image_path'].split('./eval_images/')[1])

    image = Image.open(image_path)

    image = test_transform(image)

    x = Variable(image,requires_grad=True)

    x = x.cuda(non_blocking=True).unsqueeze(0)

    y = resnet50_model(x)

    y = y.reshape(-1)[class_id]

    grad_x, = torch.autograd.grad(y, x, create_graph=create_graph)

    jacobian =grad_x.reshape(x.shape)

    print('jacobian_shape : ',jacobian.shape)

    return flask.jsonify({'jacobian':jacobian,'sample_id':sample_id,'class_id':class_id,'image_id':image_id})
    


if __name__ == '__main__':
    app.run()





