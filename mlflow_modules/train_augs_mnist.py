import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.init as init
import time
import os
from datetime import datetime
from google.colab import files

from model.vitbn import ViTBN
from emnist_digit_preprocessing import download_emnist
from emnist_digit_preprocessing import MNISTCustomDataset

import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

#from dagshub import dagshub_logger

#mlflow.set_tracking_uri("https://dagshub.com/anindyahepth/BatchNorm_in_Transformers_CV")
#os.environ['MLFLOW_TRACKING_USERNAME'] = USER_NAME
#os.environ['MLFLOW_TRACKING_PASSWORD'] = PASSWORD

#EMNIST_DATASET

def get_datasets_emnist() :
  dir_root = '/content/'
  file_dict={
    'train_images':'emnist-digits-train-images-idx3-ubyte.gz',
    'train_labels':'emnist-digits-train-labels-idx1-ubyte.gz',
    'test_images':'emnist-digits-test-images-idx3-ubyte.gz',
    'test_labels':'emnist-digits-test-labels-idx1-ubyte.gz'
  }
  dataset = download_emnist(dir_root,file_dict)

  train_images=dataset[0]
  train_labels=dataset[1]
  test_images=dataset[2]
  test_labels=dataset[3]

  data_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda img: torchvision.transforms.functional.rotate(img, -90),
    lambda img: torchvision.transforms.functional.hflip(img),
    transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
    transforms_v2.RandomZoomOut(0,(2.0, 2.0), p=0.2),
    transforms.Resize(28),
    transforms.ToTensor()
    ]
)

#training_dataset

  train_dataset = MNISTCustomDataset(train_images, train_labels, transform=data_transform, label_type='integer')

#validation_dataset

  validation_dataset = MNISTCustomDataset(test_images, test_labels, transform=data_transform, label_type='integer')

  return train_dataset, validation_dataset

#MNIST_DATASET

def get_datasets_mnist() :
  data_transform = transforms.Compose([
	transforms.RandomRotation(30),
    	transforms.RandomAffine(degrees = 0, translate = (0.3, 0.3)),
    	transforms_v2.RandomZoomOut(0,(1.0, 4.0), p=0.4),
	transforms.Resize(28),
    	transforms.ToTensor()
    ])

  #Load training data

  train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform= data_transform)

  #Load validation data
  validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform= data_transform)

  return train_dataset, validation_dataset

#FMNIST_DATASET

def get_datasets_fmnist() :
  data_transform = transforms.Compose([
	transforms.Resize(28),
    	transforms.ToTensor()
    ])
	
#Load the training dataset
  train_dataset = dsets.FashionMNIST('~/.pytorch/F_MNIST_data', train=True, download=True,transform= tensor_transform)

#Load the validation dataset

  validation_dataset = dsets.FashionMNIST('~/.pytorch/F_MNIST_data', train=False, download=True, transform= tensor_transform)

  return train_dataset, validation_dataset
	


def train_model(model,train_loader,validation_loader, train_dataset, validation_dataset, optimizer, scheduler,criterion,n_epochs):

    #global variable
    N_test=len(validation_dataset)	
    N_train=len(train_dataset)
    accuracy_train_list=[]
    accuracy_list=[]
    cost_list=[]
    cost_test_list=[]
    loss_list=[]
    dur_list_train=[]
    dur_list_val = []
    class_correct = [[0.for i in range(10)] for j in range(n_epochs)]
    class_total= [[0.for i in range (10)] for j in range(n_epochs)]
    class_accuracy =[[0.for i in range(10)] for j in range(n_epochs)]
    COST=0.
    COST_test=0.
    correct_train=0.
    correct=0.
    delta_train=0
    delta_val=0
    train_time=0
    test_time=0

    for epoch in range(n_epochs):
        COST=0.
        correct_train=0
        t0 = datetime.now()
        for x, y in train_loader:
            optimizer.zero_grad()
            model.train()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST+=loss.data.item()
            _, yhat = torch.max(z, 1)
            correct_train += (yhat == y).sum().item()
            #loss_list.append(loss.data)
        scheduler.step()
        cost_list.append(COST)
        accuracy_train= correct_train/N_train
        accuracy_train_list.append(accuracy_train)
        delta_train= datetime.now() - t0
        dur_list_train.append(delta_train.total_seconds())

        correct = 0.
        COST_test=0.
        t1 = datetime.now()
        #perform a prediction on the validation  data
        for x_test, y_test in validation_loader:
            model.eval()
            z = model(x_test)
            _, yhat = torch.max(z, 1)
            correct += (yhat == y_test).sum().item()
            loss_test = criterion(z, y_test)
            COST_test +=loss_test.data.item() 
        cost_test_list.append(COST_test)
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
        delta_val=datetime.now() - t1
        dur_list_val.append(delta_val.total_seconds())

    train_time = sum(dur_list_train)/n_epochs
    test_time = sum(dur_list_val)/n_epochs
    fin_lr = scheduler.optimizer.param_groups[0]["lr"]

    return cost_list, accuracy_list, dur_list_train, dur_list_val, accuracy_train_list, cost_test_list, train_time, test_time, fin_lr





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
                pos_emb ='learn'
    )

  #model.load_state_dict(torch.load("model.pth"))

  return model





if __name__ == "__main__":

    mlflow.pytorch.autolog()


    learning_rate = 0.00063
    n_epochs = 50
    batch_size = 20
    gamma = 0.24
    step_size = 1
    criterion = nn.CrossEntropyLoss()
    

    with mlflow.start_run(experiment_id=49):
        train_dataset, validation_dataset = get_datasets_mnist()
        model = get_model()
        #logged_model = 'runs:/6f43f730540040b1a28fbcce66b40dc3/model_mnist'
        #model = mlflow.pytorch.load_model(logged_model)
        
        
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size= step_size,gamma= gamma)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
        validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=5000, shuffle=True)
        
        
        cost_list, accuracy_list, dur_list_train, dur_list_val, accuracy_train_list, cost_test_list, train_time, test_time, fin_lr = train_model(model,train_loader,validation_loader, train_dataset, validation_dataset,optimizer, scheduler, criterion,n_epochs)

		

      #  with dagshub_logger() as logger:
      #      logger.log_metrics(loss_tr=cost_list, accuracy_val=accuracy_list, time_tr = dur_list_train, time_val=dur_list_val)
      #     logger.log_hyperparams({
      #         "learning_rate": learning_rate,
      #         "epochs": n_epochs
      #      })
      
      
        
        mlflow.log_params({
            "learning_rate": learning_rate,
	    "batch_size" : batch_size,
            "epochs": n_epochs,
            "decay_rate": gamma,
	    "step_size" : step_size
        })
        
        #mlflow.pytorch.log_model(model,"model_mnist")
        
        mlflow.log_metrics(
            {
              "avg_train_time": train_time,
              "average_test_time" : test_time,
	      "final_lr" : fin_lr

            })

        for i in range(n_epochs):
        	mlflow.log_metrics(
            {
                "training_loss": cost_list[i],
                "validation_accuracy": accuracy_list[i],
                "training_time": dur_list_train[i],
                "validation_time": dur_list_val[i],
                "training_accuracy": accuracy_train_list[i],
	        "test_loss": cost_test_list[i]
            },step = i+1
        )

        #print("Saving the model...")
        
torch.save(model.state_dict(), 'model.pth')

# download checkpoint file

files.download('model.pth')     

print("done.")
print("fin_lr:", fin_lr)
