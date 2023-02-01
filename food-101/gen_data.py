import os
import shutil
import pandas as pd


train_dir="train"
os.makedirs(train_dir)
with open("finetune_food101_train.csv") as f:
    for i,line in enumerate(f):
        print(i,line)
        if i >0:
            line=line.strip()
            line=line.split(",")
            src_path=line[1]
            cat_idx=line[2]
            des_dir=os.path.join(train_dir,cat_idx)
            print('src_path,des_dir : ',src_path,des_dir)
            if not os.path.exists(des_dir):
                os.makedirs(des_dir)
            shutil.copy(src_path, des_dir)


test_dir="val"
os.makedirs(test_dir)
with open("finetune_food101_test.csv") as f:
    for i,line in enumerate(f):
        print(i,line)
        if i>0:            
            line=line.strip()
            line=line.split(",")
            src_path=line[1]
            cat_idx=line[2]
            des_dir=os.path.join(test_dir,cat_idx)
            if not os.path.exists(des_dir):
                os.makedirs(des_dir)
            shutil.copy(src_path, des_dir)
