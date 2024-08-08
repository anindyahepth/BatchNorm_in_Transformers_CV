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

from model.vitbn import ViTBN
from emnist_digit_preprocessing import download_emnist
from emnist_digit_preprocessing import MNISTCustomDataset

import mlflow

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))




def get_datasets() :
  dir_root = '/content/'
  file_dict={
    'test_images':'emnist-digits-test-images-idx3-ubyte.gz',
    'test_labels':'emnist-digits-test-labels-idx1-ubyte.gz'
  }
  dataset = download_emnist(dir_root,file_dict)
  return dataset
  
def get_infdata(images, labels) : 
  data_transform = transforms.Compose([
    transforms.ToPILImage(),
    lambda img: torchvision.transforms.functional.rotate(img, -90),
    lambda img: torchvision.transforms.functional.hflip(img),
    transforms.Resize(28),
    transforms.ToTensor()
    ]
)

  inference_dataset = MNISTCustomDataset(images, labels, transform=data_transform, label_type='integer')

  return inference_dataset



def eval_model(model,validation_dataset, validation_loader,n_epochs):

    #global variable
    N_test=len(validation_dataset)
    accuracy_list=[]
    dur_list_val = []
    delta_val =0
    correct=0
    # class_correct = [[0.for i in range(10)] for j in range(n_epochs)]
    # class_total= [[0.for i in range (10)] for j in range(n_epochs)]
    # class_accuracy =[[0.for i in range(10)] for j in range(n_epochs)]
    
  
    #perform a prediction on the validation  data
    for epoch in range(n_epochs):
        t1 = datetime.now()
        correct=0
        for x_test, y_test in validation_loader:
            model.eval()
            z = model(x_test)
            _, yhat = torch.max(z, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)
        delta_val=datetime.now() - t1
        dur_list_val.append(delta_val.total_seconds())

    return accuracy_list, dur_list_val
  

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

  model.load_state_dict(torch.load("ViTBN_30_mnist.pth"))

  return model



if __name__ == "__main__":

    mlflow.pytorch.autolog()

    n_epochs = 10
    batch_size = 100
    

    with mlflow.start_run(experiment_id=17):
        dataset = get_datasets()
        test_images = dataset[0]
        test_labels = dataset[1]
        inference_dataset = get_infdata(test_images, test_labels)
        model = get_model()
        #logged_model = 'runs:/6f43f730540040b1a28fbcce66b40dc3/model_mnist'
        #model = mlflow.pytorch.load_model(logged_model)
        
        inference_loader = torch.utils.data.DataLoader(dataset=inference_dataset, batch_size=100, shuffle=True)
        
        
        accuracy_list, dur_list_val = eval_model(model, inference_dataset, inference_loader,n_epochs)

        mlflow.log_params({
            "epochs": n_epochs,
            "batch_size": batch_size
        })
        
        mlflow.log_metrics({
            "average_inf_time" : sum(dur_list_val)/n_epochs

       })
        
        for i in range(n_epochs):
        	mlflow.log_metrics(
            {
                "inference_accuracy": accuracy_list[i],
                "inference_time": dur_list_val[i]
            },step = i+1
        )

            

print("done.")

