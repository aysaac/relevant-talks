import os
from PIL import Image
import wandb
import matplotlib.pyplot as plt
import numpy as np
from mpmath import degree
from torch.nn.functional import relu
from sklearn.metrics import f1_score,precision_score,recall_score
from torch import nn, flatten
from torch.utils.data import Dataset,DataLoader
from models import SimpleClassifier
import torch


class DirectoryDataset(Dataset):

    def __init__(self, directory:str):

        self.dir=directory
        self.image_paths=os.listdir(directory)
        self.targets=[image_path.split("_")[1] for image_path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        target=self.targets[idx]

        image=~np.array(Image.open(self.image_paths[idx]))

        return image,target

model_path=r"C:\Users\isaac\PycharmProjects\creative_clustering\wandb_examples\mnist_cnn_swept-pine-15.pt"

model=SimpleClassifier()

model.load_state_dict(torch.load(model_path))

wandb.init(project="mnist-example",
           config={"batch_size": 2000,
                   "test_batch_size": 1000,
                   "epochs":10,
                   "lr": 0.001,
                   "weight_decay": 0.01,
                   "no_cuda": False,
                   "seed": 42,
                   "log_interval": 100}
           )
wandb.name= "testing-"+ str(wandb.run.name)

