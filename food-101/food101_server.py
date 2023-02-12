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
from torch.autograd import Variable
from base64 import b64encode


base_dir = '/home/zhaoy32/Desktop/swa_gaussian/food-101/'
image_la_result_dir = os.path.join(base_dir,'image_la_result/')
la_all_images_result_dir = os.path.join(base_dir,'la_test_result')
la_sample_model_dir = os.path.join(base_dir,'googlenet_la_sample_models') 

# base_dir = base_dir if base_dir[-1] == '/' else base_dir+'/'

model_ft = torch.load(os.path.join(base_dir,"food101_googlenet_finetune.pt")) 

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

CHOOSED_CLASSES = ['french_toast', 'greek_salad', 'caprese_salad', 'chocolate_cake', 'pizza', 'cup_cakes', 'carrot_cake','cheesecake','pancakes', 'strawberry_shortcake']


app = Flask(__name__)
CORS(app)

@app.route('/calculate_jacobian', methods=['GET','POST'])

def calculate_jacobian():

    client_data = flask.request.json
    image_id = client_data['image_id']
    sample_id = client_data['sample_id']
    class_id = client_data['class_id']
    sample_model_dir = os.path.join(la_sample_model_dir,'la_googlenet_sample_'+str(sample_id)+'.pt')
    checkpoint = torch.load(sample_model_dir)
    # vgg16_model.load_state_dict(checkpoint["state_dict"])
    model_ft.load_state_dict(checkpoint) # we saved the state_dict of each sample googlenet model as .pt  
    model_ft.eval()
    image_path = os.path.join(base_dir, 'images', image_id_path_dict[str(image_id)]['image_path'].split('./eval_images/')[1])
    image = Image.open(image_path)
    image = test_transform(image)
    x = image.cuda(non_blocking=True).unsqueeze(0)
    x.requires_grad_(True)
    y = model_ft(x) # shape (1,10) tensor([[ -3.0459,  -9.2013,  -8.5121,   3.3400, -12.8477,  -6.3842,  -3.1321, -3.0611,  -0.2655,  -3.6536]], device='cuda:0', grad_fn=<AddmmBackward0>)
    predict_id = torch.argmax(y[0])
    y_class = y[0][class_id] #(1,10)
    if x.grad is not None:
        x.grad.data.fill_(0)       
    y_class.backward(retain_graph=True) #(1,3,224,224)
    print('grad_x shape :',x.grad.data.shape) #(1,3,224,224)
    jacobian = x.grad.data[0].cpu().data.numpy()
    print('jacobian shape : ',jacobian.shape) #(3,224,224)
 
    return flask.jsonify({'jacobian': b64encode(jacobian.tobytes()).decode(), 'jacobian_shape': jacobian.shape,'sample_id':sample_id,'class_id':class_id,'image_id':image_id,'prediction': predict_id.item()})
    


@app.route('/calculate_jacobian_finetune', methods=['GET','POST'])

def calculate_jacobian_finetune():

    client_data = flask.request.json
    image_id = client_data['image_id']
    class_id = client_data['class_id']
    
    image_path = os.path.join(base_dir, 'images', image_id_path_dict[str(image_id)]['image_path'].split('./eval_images/')[1])
    image = Image.open(image_path)
    image = test_transform(image)
    x = image.cuda(non_blocking=True).unsqueeze(0)
    x.requires_grad_(True)
    y = model_ft(x) # shape (1,10) tensor([[ -3.0459,  -9.2013,  -8.5121,   3.3400, -12.8477,  -6.3842,  -3.1321, -3.0611,  -0.2655,  -3.6536]], device='cuda:0', grad_fn=<AddmmBackward0>)
    predict_id = torch.argmax(y[0])
    y_class = y[0][class_id] #(1,10)
    if x.grad is not None:
        x.grad.data.fill_(0)       
    y_class.backward(retain_graph=True) #(1,3,224,224)
    print('grad_x shape :',x.grad.data.shape) #(1,3,224,224)
    jacobian = x.grad.data[0].cpu().data.numpy()
    print('jacobian shape : ',jacobian.shape) #(3,224,224)
 
    return flask.jsonify({'jacobian': b64encode(jacobian.tobytes()).decode(), 'jacobian_shape': jacobian.shape,'class_id':class_id,'image_id':image_id,'prediction': predict_id.item()})
    

# @app.route('/calculate_jacobian', methods=['GET','POST'])

# def calculate_jacobian():

#     client_data = flask.request.json
#     image_id = client_data['image_id']
#     sample_id = client_data['sample_id']
#     class_id = client_data['class_id']
#     sample_model_dir = os.path.join(la_sample_model_dir,'la_googlenet_sample_'+str(sample_id)+'.pt')
#     checkpoint = torch.load(sample_model_dir)
#     # vgg16_model.load_state_dict(checkpoint["state_dict"])
#     model_ft.load_state_dict(checkpoint) # we saved the state_dict of each sample googlenet model as .pt  
#     model_ft.eval()
#     image_path = os.path.join(base_dir, 'images', image_id_path_dict[str(image_id)]['image_path'].split('./eval_images/')[1])
#     image = Image.open(image_path)
#     image = test_transform(image)
#     x = Variable(image,requires_grad=True)
#     x = x.cuda(non_blocking=True).unsqueeze(0)
#     y = model_ft(x)
#     predict_id = torch.argmax(y[0])
#     y = y[0][class_id] #(1,10)
#     grad_x, = torch.autograd.grad(y, x, create_graph=True)
#     jacobian = grad_x.reshape(x.shape).data[0].cpu().data.numpy() #(1,3,224,224)
#     print('x shape : ',x.shape)
#     return flask.jsonify({'jacobian': b64encode(jacobian.tobytes()).decode(), 'jacobian_shape': jacobian.shape,'sample_id':sample_id,'class_id':class_id,'image_id':image_id,'prediction': predict_id.item()})
   


@app.route('/image_la_kl', methods=['GET','POST'])

def get_image_la_kl():
    # client_data = flask.request.json
    # image_id = client_data['image_id']
    data = json.load(open(base_dir+'image_la_kl.json','r'))
    return flask.jsonify({'data':data})


@app.route('/image_la_result', methods=['GET','POST'])

def get_image_la_result():
    client_data = flask.request.json
    image_id = client_data['image_id']
    data = json.load(open(image_la_result_dir+'image_'+str(image_id)+'.json','r'))
    return flask.jsonify({'image_id':image_id,'data':data})



@app.route('/la_all_images_result', methods=['GET','POST'])

def get_la_all_images_result():
    client_data = flask.request.json
    sample_id = client_data['sample_id']
    data = json.load(open(os.path.join(la_all_images_result_dir,'la_test_sample_'+str(sample_id)+'.json'),'r')) # for googlenet data
    # data = json.load(open(la_all_images_result_dir+'result_dict_'+str(sample_id)+'.json','r')) #for resnet and vgg data
    return flask.jsonify({'sample_id':sample_id,'data':data})
    



@app.route('/calculate_jacobian_avg', methods=['GET','POST'])

def calculate_jacobian_avg():

    client_data = flask.request.json
    image_id = client_data['image_id']
    data = json.load(open(os.path.join(base_dir,'jac_avg','jac_avg_'+str(image_id)+'.json'),'r'))
    for i in range(len(data)): # class id
        data[i] = {'label_id': i, 'label': CHOOSED_CLASSES[i] ,'data': data[i]}
    return flask.jsonify({'jacobian_avg':data,'image_id':image_id})
    


if __name__ == '__main__':
    app.run()





