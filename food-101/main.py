import torch, os, torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.utils import save_image
from PIL import Image
from glob import glob
from tqdm import tqdm
import json

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

IMG_SIZE = (224, 224) 
BATCH_SIZE = 1 
BATCH_SIZE_TRAIN = 8
IMG_MEAN = [0.485, 0.456, 0.406]
IMG_STD = [0.229, 0.224, 0.225]
CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if CUDA else "cpu")

BASE_PATH = './'
TEST_SPLIT = 0.1
epochs = 20


# select classes in the given dataset 

Datasets = 'FOOD101'
# # Datasets = 'pascal voc'

# all_img_df = pd.DataFrame({'path': glob(os.path.join(BASE_PATH, 'images', '*', '*.jpg'))})
# all_img_df['category'] = all_img_df['path'].map(lambda x: os.path.split(os.path.dirname(x))[-1])
# all_img_df['image_id'] = all_img_df.index

# cat_enc = LabelEncoder()
# all_img_df['cat_idx'] = cat_enc.fit_transform(all_img_df['category'])
# N_CLASSES = len(cat_enc.classes_)

# print(N_CLASSES, 'classes')
# print(set(all_img_df['category'].tolist()))

if Datasets == 'FOOD101':
    CHOOSED_CLASSES = ['french_toast', 'greek_salad', 'caprese_salad', 'chocolate_cake', 'pizza', 'cup_cakes', 'carrot_cake','cheesecake','pancakes', 'strawberry_shortcake']
elif Datasets == 'pascal voc':
    CHOOSED_CLASSES = []

# mask = all_img_df['category'].isin(CHOOSED_CLASSES)
# all_img_df_selected = all_img_df.loc[mask]


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
            #transforms.Normalize(IMG_MEAN, IMG_STD)
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
        
        return image, int(target),image_path,image_id

    def __len__(self):
        return self.dataset.shape[0]


# train_df, test_df = train_test_split(all_img_df_selected,
#                                      test_size=TEST_SPLIT,
#                                      random_state=42,
#                                      stratify=all_img_df_selected['category'])

# print('train', train_df.shape[0], 'test', test_df.shape[0])

# train_df.to_csv('finetune_food101_train.csv')
# test_df.to_csv('finetune_food101_test.csv')

train_df = pd.read_csv('finetune_food101_train.csv')
test_df = pd.read_csv('finetune_food101_test.csv')

train_dataset = myDataset(train_df, 'train')
train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                batch_size=BATCH_SIZE_TRAIN, pin_memory=False)#, num_workers=4)

test_dataset = myDataset(test_df, 'test')
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False,
                batch_size=BATCH_SIZE, pin_memory=False)#, num_workers=1)

model_ft = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
# model_ft = models.resnet50(pretrained=True) 

for param in model_ft.parameters():
    param.requires_grad = False
print(model_ft.fc)

num_fc_ftr = model_ft.fc.in_features
print(num_fc_ftr)

model_ft.fc = nn.Linear(num_fc_ftr, len(CHOOSED_CLASSES))
print(model_ft.fc)

model_ft = model_ft.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam([{'params': model_ft.fc.parameters()}], lr=0.001)

def train(model,device, train_loader, epoch):
    model.train()
    for batch_idx, data in enumerate(tqdm(train_loader)):
        image,label_id,image_path,image_id = data
        image=image.to(device)
        label_id=label_id.to(device)
        optimizer.zero_grad()
        y_hat= model(image)
        loss = criterion(y_hat, label_id)
        loss.backward()
        optimizer.step()
    print ('Train Epoch: {}\t Loss: {:.6f}'.format(epoch, loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    softmax_result_list = []

    image_id_path_test_dict = {}
    image_path_id_test_dict = {}

    with torch.no_grad():

        for i,data in enumerate(tqdm(test_loader)):

            image,label_id,image_path,image_id = data

            eval_image_path = "./eval_images"+image_path[0].split('./images')[1]

            save_image(image[0],eval_image_path)

            transform_norm = transforms.Compose([
                transforms.Normalize(IMG_MEAN, IMG_STD)
            ])
            
            image = transform_norm(image)

            image=image.to(device)
            label_id=label_id.to(device)
            optimizer.zero_grad()
            y_hat = model(image)
            #print('y_hat shape : ',y_hat.shape) #(batch size, num of classes)
            y_softmax = F.softmax(y_hat,dim=1)
            #print('y_softmax : ',y_softmax.shape,y_softmax) #(batch size, num of classes)
            #print('image_path, image_id',image_path,image_id)
            image_id_path_test_dict[i]=eval_image_path
            image_path_id_test_dict[eval_image_path]=i
            softmax_result_list.append({'image_path':eval_image_path,'image_id':int(image_id.item()),'label_id':int(label_id.item()),'label':CHOOSED_CLASSES[label_id],'predict_softmax':y_softmax[0].tolist()})
            # print(softmax_result_list)
            test_loss += criterion(y_hat, label_id).item() # sum up batch loss
            pred = y_hat.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(label_id.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_dataset),
        100. * correct / len(test_dataset)))

    json.dump(image_id_path_test_dict,open('./image_id_path_test_dict.json','w'))
    json.dump(image_path_id_test_dict,open('./image_path_id_test_dict.json','w'))

    return softmax_result_list


for epoch in range(epochs):
    train(model=model_ft,device=DEVICE, train_loader=train_loader, epoch=epoch)
    softmax_result_list = test(model=model_ft, device=DEVICE, test_loader=test_loader)
    
torch.save(model_ft,'food101_googlenet_finetune.pt')

json.dump(softmax_result_list,open('./food101_googlenet_finetune_test_result.json','w'))






