import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.init as init
import time
import os
from datetime import datetime
from google.colab import files

from model.vitbnv4 import ViTBN

import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

#from dagshub import dagshub_logger

#mlflow.set_tracking_uri("https://dagshub.com/anindyahepth/BatchNorm_in_Transformers_CV")
#os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
#os.environ['MLFLOW_TRACKING_PASSWORD'] = PASSWORD




def get_datasets() :
  data_transform = transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor()
    ])

  #Load training data

  train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform= data_transform)

  #Load validation data
  #validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform= data_transform)

  return train_dataset


def train_model(model,train_loader, optimizer,criterion,n_epochs):

    #global variable
    cost_list=[]
    dur_list_train=[]
    delta_train=0

    for epoch in range(n_epochs):
        COST=0.
        t0 = datetime.now()
        for x, y in train_loader:
            optimizer.zero_grad()
            model.train()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST+=loss.data.item()
            #loss_list.append(loss.data)
        cost_list.append(COST)
        delta_train= datetime.now() - t0
        dur_list_train.append(delta_train.total_seconds())

        

    return cost_list, dur_list_train





def get_model():
  model = ViTBN(
                image_size = 28,
                patch_size = 7,
                num_classes = 10,
                channels =1,
                dim = 64,
                depth = 6,
                heads = 8,
                mlp_dim = 128,
                pool = 'cls',
                dropout = 0.0,
                emb_dropout = 0.0,
                pos_emb = 'learn'
    )

  #model.load_state_dict(torch.load("model100epoch_mnist.pth"))

  return model





if __name__ == "__main__":

    mlflow.pytorch.autolog()


    learning_rate = 0.0005
    n_epochs = 30
    criterion = nn.CrossEntropyLoss()
    

    with mlflow.start_run(experiment_id=21):
        train_dataset = get_datasets()
        model = get_model()
        #logged_model = 'runs:/6f43f730540040b1a28fbcce66b40dc3/model_mnist'
        #model = mlflow.pytorch.load_model(logged_model)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100,shuffle=True)
        #validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=True)
        
        
        cost_list, dur_list_train = train_model(model,train_loader,optimizer,criterion,n_epochs)

		

      #  with dagshub_logger() as logger:
      #      logger.log_metrics(loss_tr=cost_list, accuracy_val=accuracy_list, time_tr = dur_list_train, time_val=dur_list_val)
      #     logger.log_hyperparams({
      #         "learning_rate": learning_rate,
      #         "epochs": n_epochs
      #      })
      
      
        
        mlflow.log_params({
            "learning_rate": learning_rate,
            "epochs": n_epochs
        })
        
        mlflow.pytorch.log_model(model,"model_mnist")
        
        for i in range(n_epochs):
        	mlflow.log_metrics(
            {
                "training_loss": cost_list[i],
                "training_time": dur_list_train[i]
            },step = i+1
        )

        #print("Saving the model...")
        
#torch.save(model.state_dict(), 'model_mnist.pth')

# download checkpoint file

#files.download('model_mnist.pth')     

print("done.")
