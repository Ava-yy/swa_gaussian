import torch, os, torchvision
import random
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from torch.distributions import MultivariateNormal
from PIL import Image
from glob import glob
from tqdm import tqdm
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal

from laplace import Laplace

IMG_SIZE = (224, 224) 
BATCH_SIZE = 1 
BATCH_SIZE_TRAIN = 8
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")
# DEVICE = torch.device("cpu")
Datasets = 'FOOD101'
NUM_MODELS = 100

BASE_DIR = '/home/zhaoy32/Desktop/swa_gaussian/food-101'
LA_MODELS_DIR = os.path.join(BASE_DIR,'googlenet_la_sample_models')
LA_TEST_DIR = os.path.join(BASE_DIR,'la_test_result')

if Datasets == 'FOOD101':
    CHOOSED_CLASSES = ['french_toast', 'greek_salad', 'caprese_salad', 'chocolate_cake', 'pizza', 'cup_cakes', 'carrot_cake','cheesecake','pancakes', 'strawberry_shortcake']

BASE_PATH = './'
TEST_SPLIT = 0.1
epochs = 20


class myDataset(torch.utils.data.Dataset):

    def __init__(self, image_df, mode='train', CHOOSED_CLASSES=CHOOSED_CLASSES):
        self.dataset = image_df
        self.CHOOSED_CLASSES = CHOOSED_CLASSES
        self.mode = mode

        train_transforms = transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMG_MEAN, IMG_STD)
        ])

        if mode == 'train':
            self.transform = train_transforms
        else:
            self.transform = val_transforms

    def __getitem__(self, index):

        c_row = self.dataset.iloc[index]

        image_id = c_row['image_id']

        image_path, target = c_row['path'], self.CHOOSED_CLASSES.index(c_row['category'])  #image and target
        image = Image.open(image_path)

        image = self.transform(image)
        
        return image, int(target),c_row['category'], image_path,image_id

        # return image, int(target)


    def __len__(self):
        return self.dataset.shape[0]


def predict(loader, model, sample_id, cuda=True, verbose=False):
    predictions = list()
    targets = list()

    model.eval()
    print(model)
    prediction_list = []
    
    with torch.no_grad():
        # for loader_idx, (input, target) in enumerate(loader): # start from category 0
        for i, (input, target, category, image_path,image_id) in enumerate(loader):
            if cuda:
                input = input.cuda(non_blocking=True)
            output = model(input)
            batch_size = input.size(0)
            #print('batch size : ',batch_size) 1
            prediction = F.softmax(output, dim=1).cpu().numpy()
            prediction_list.append({'image_path':image_path[0],'label_id':int(target.item()),'label':category[0],'image_id':int(image_id.item()),'predict_softmax':prediction[0].tolist(),'sample_id':sample_id})
            targets.append(target.numpy())
            predictions.append(prediction)

    return {"result_dict": prediction_list, "predictions": np.vstack(predictions),"targets": np.concatenate(targets)}


criterion = nn.CrossEntropyLoss()

train_df = pd.read_csv('finetune_food101_train.csv')
test_df = pd.read_csv('finetune_food101_test.csv')

train_dataset = myDataset(train_df, 'train')
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                batch_size=BATCH_SIZE_TRAIN, pin_memory=False)#, num_workers=4)

test_dataset = myDataset(test_df, 'test')
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                batch_size=BATCH_SIZE, pin_memory=False)#, num_workers=1)

# load pretrained resnet model
# model_ft = torch.load("food101_finetune.pt")

# load fintuned googlenet model
model_ft = torch.load("food101_googlenet_finetune.pt")

model_ft.to(DEVICE)
print(model_ft)
model_ft.eval()

print('start laplace')
la = Laplace(model_ft, 'classification',
             subset_of_weights='last_layer',
             hessian_structure='full')

la.fit(train_loader)

Hessian = la.posterior_covariance
print('Hessian :', Hessian.size()) #(20490,20490) for resnet, (10240,10240) for googlenet

Hessian = Hessian.detach().cpu().numpy()
cov_mat_fc = inv(Hessian[:10240,:10240])

mean_fc = model_ft.state_dict()['fc.weight'] # (num of classes in food101, 2048) for googlenet; (10,4096) for vgg16; (10,2048) for resnet
mean_fc = mean_fc.detach().cpu().numpy()

np.save(open(os.path.join(BASE_DIR,'googlenet_fc_mean.npy'),'wb'),mean_fc)
np.save(open(os.path.join(BASE_DIR,'googlenet_fc_cov_mat.npy'),'wb'),cov_mat_fc)

mean_fc = np.load(open(os.path.join(BASE_DIR,'googlenet_fc_mean.npy'),'rb'))
cov_mat_fc = np.load(open(os.path.join(BASE_DIR,'googlenet_fc_cov_mat.npy'),'rb'))

print('mean_fc shape :', mean_fc.shape)
print('cov_mat_fc shape : ',cov_mat_fc.shape)

for sample_id in range(NUM_MODELS):

    print('sample id : ',sample_id)

    random.seed(sample_id)

    # pytorch method
    # m = MultivariateNormal(torch.tensor(mean_fc), torch.tensor(cov_mat_fc)) # https://pytorch.org/docs/stable/distributions.html
    # fc_weight_sample = m.sample() # TODO: didn't find where to set seed  https://pytorch.org/docs/stable/_modules/torch/distributions/distribution.html#Distribution.sample
    # fc_weight_sample = fc_weight_sample.numpy()

    # numpy method
    fc_weight_sample = np.random.multivariate_normal(mean_fc.flatten(),cov_mat_fc) #

    # scipy method
    # fc_weight_sample = multivariate_normal.rvs(mean=mean_fc.flatten(), cov=cov_mat_fc, size=1, random_state=sample_id)

    print('fc_weight_la shape: ',fc_weight_sample.shape) #(10240,)
    with torch.no_grad():  
        for name, param in model_ft.named_parameters():
            print(name,param.numel(),param.size())
            if name=='fc.weight':
                print('before param', param,param.shape)
                param.data = torch.tensor(fc_weight_sample.reshape(param.shape)) #(num of classes in food101, 2048) for googlenet; (10,4096) for vgg16; (10,2048) for resnet
                print('after param : ',param,param.shape)

    torch.save(model_ft.state_dict(), os.path.join(LA_MODELS_DIR,'la_googlenet_sample_'+str(sample_id)+'.pt'))

    # test sample model on test dataset
    model_ft = torch.load("food101_googlenet_finetune.pt")
    model_ft.load_state_dict(torch.load(os.path.join(LA_MODELS_DIR,'la_googlenet_sample_'+str(sample_id)+'.pt')))
    res = predict(test_loader, model_ft, sample_id=sample_id, verbose=True)
    prediction_result_dict = res["result_dict"]
    json.dump(prediction_result_dict,open(os.path.join(LA_TEST_DIR, 'la_test_sample_'+str(sample_id)+'.json'),'w'))









