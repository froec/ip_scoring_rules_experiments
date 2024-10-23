import numpy as np 
import pandas as pd
import os
import argparse 

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from PIL import Image

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Parse arguments for the model.")
    parser.add_argument("--datadir", type=str, default="data/celeba-kaggle/", help="Data directory (containg the relevant accompanying csv files).")
    parser.add_argument("--imgdir", type=str, default="data/celeba-kaggle/img_align_celeba/img_align_celeba/", help="Data directory.")
    parser.add_argument("--outdir", type=str, default="data/celeba_embeddings/", help="Output directory (where to save the embeddings, which are then used for training).")
    # note that the created csv files are saves in the --datadir
    parser.add_argument("--batch_size", type=int, default=64, help="Size of mini-batches")
    return parser.parse_args()

args = parse_args()



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: %s" % device)



######################################################
"""
We read the csv files accompanying the CelebA datasets and split into train/val/test
Also split based on the group (here: gender)
"""
######################################################
createTrainValTestCSVs = True
if createTrainValTestCSVs:
    # load the data
    nRowsRead = None 
    attr_df = pd.read_csv(args.datadir + "list_attr_celeba.csv", nrows = nRowsRead)
    attr_df.dataframeName = "list_attr_celeba.csv"

    # "partition"=0 is train, "partition"=1 is validation, "partition"=2 is test
    partition_df = pd.read_csv(args.datadir + "list_eval_partition.csv", nrows = nRowsRead)
    partition_df.dataframeName = "list_eval_partition.csv"
    print(partition_df.head())


    info_df = attr_df.merge(partition_df, left_index=True, right_index=True)
    print(info_df.head())


    info_df.loc[info_df["partition"] == 0].to_csv(args.datadir +"celeba-train.csv")
    info_df.loc[info_df["partition"] == 1].to_csv(args.datadir +"celeba-val.csv")
    info_df.loc[info_df["partition"] == 2].to_csv(args.datadir +"celeba-test.csv")


    # save with gendered groups
    # there is male, female, uncertain in the data
    malemask = (info_df["Male"]==1.)
    print(malemask.sum())

    info_df.loc[((info_df["partition"]) == 0) & malemask].to_csv(args.datadir +"celeba-train-male.csv")
    info_df.loc[((info_df["partition"]) == 0) & (~malemask)].to_csv(args.datadir +"celeba-train-nonmale.csv")


    info_df.loc[((info_df["partition"]) == 1) & malemask].to_csv(args.datadir +"celeba-val-male.csv")
    info_df.loc[((info_df["partition"]) == 1) & ~malemask].to_csv(args.datadir +"celeba-val-nonmale.csv")

    info_df.loc[((info_df["partition"]) == 2) & malemask].to_csv(args.datadir +"celeba-test-male.csv")
    info_df.loc[((info_df["partition"]) == 2) & ~malemask].to_csv(args.datadir +"celeba-test-nonmale.csv")
    
    # smaller train set
    info_df.loc[((info_df["partition"]) == 0)].head(1000).to_csv(args.datadir +"celeba-train-small.csv")
    
    # smaller test set
    info_df.loc[((info_df["partition"]) == 2)].head(1000).to_csv(args.datadir +"celeba-test-small.csv")
    
    

    print("saved training, validation, test splits as csv files")




class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA"""
    # inspired by https://github.com/rasbt/deeplearning-models/blob/master/pytorch_ipynb/cnn/cnn-resnet18-celeba-dataparallel.ipynb

    def __init__(self, csv_path, img_dir, transform=None, subsample=None):
    
        df = pd.read_csv(csv_path)
        ### subsample
        if subsample is not None:
            df = df.sample(n=subsample)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df["image_id_x"].values

        ### the target label:
        self.y = df["Wearing_Earrings"].values
        self.y[self.y==-1.] = 0.
        self.y[self.y==1.] = 1.


        self.transform = transform

        self.classes = [0.,1.] # binary task

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]




######################################################
"""
We are using a pre-trained Resnet50 for the embeddings
"""
######################################################
# Load pre-trained
weights = ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)
print(model)

# we should use the corresponding transform!
preprocess = weights.transforms()


######################################################
"""
Define datasets and dataloaders
"""
######################################################

### datasets
train_dataset = CelebaDataset(csv_path='celeba-train.csv',
                                  img_dir=args.imgdir,
                                  transform=preprocess)

train_male_dataset = CelebaDataset(csv_path='celeba-train-male.csv',
                                  img_dir=args.imgdir,
                                  transform=preprocess)

train_nonmale_dataset = CelebaDataset(csv_path='celeba-train-nonmale.csv',
                                  img_dir=args.imgdir,
                                  transform=preprocess)

val_dataset = CelebaDataset(csv_path='celeba-val.csv',
                              img_dir=args.imgdir,
                              transform=preprocess)

test_dataset = CelebaDataset(csv_path='celeba-test.csv',
                             img_dir=args.imgdir,
                             transform=preprocess)

test_male_dataset = CelebaDataset(csv_path='celeba-test-male.csv',
                             img_dir=args.imgdir,
                             transform=preprocess)

test_nonmale_dataset = CelebaDataset(csv_path='celeba-test-nonmale.csv',
                             img_dir=args.imgdir,
                             transform=preprocess)



### data loaders
train_loader = DataLoader(dataset=train_dataset,
                             batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0)

train_male_loader = DataLoader(dataset=train_male_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0)

train_nonmale_loader = DataLoader(dataset=train_nonmale_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0)


val_loader = DataLoader(dataset=val_dataset,
                      batch_size=args.batch_size,
                      shuffle=False,
                      num_workers=4)



test_loader = DataLoader(dataset=test_dataset,
                     batch_size=args.batch_size,
                     shuffle=False,
                     num_workers=4)

test_male_loader = DataLoader(dataset=test_male_dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=0)


test_nonmale_loader = DataLoader(dataset=test_nonmale_dataset,
                         batch_size=args.batch_size,
                         shuffle=False,
                         num_workers=0)


######################################################
"""
Now we compute the embeddings
"""
######################################################

model.fc = nn.Identity()

# as of now, we don't use the validation data
# would add: ("val",val_loader) to the list
for name, loader in [("train_male", train_male_loader), ("train_nonmale",train_nonmale_loader),\
                     ("test_male", test_male_loader), ("test_nonmale",test_nonmale_loader)]:
    print("computing embeddings for %s" % name)


    all_features = []
    all_labels = []

    with torch.no_grad():
        i_batch = 0
        for inputs, labels in loader:
            print("new batch")
            features = model(inputs)
            print(features.shape)
            all_features.append(features)
            all_labels.append(labels)
            if i_batch>499:
                break
            i_batch+=1
            
    all_features = torch.vstack(all_features)
    print(all_features.shape)
    all_labels = torch.hstack(all_labels)
    print(all_labels.shape)


    X = all_features.cpu().numpy()
    y = np.array(all_labels.cpu().numpy().flatten())

    with open(args.outdir + 'embeddings_%s.npy'%name, 'wb') as f:
        np.save(f, X)
    with open(args.outdir + 'labels_%s.npy'%name, 'wb') as f:
        np.save(f, y)
    print("saved embeddings")
    