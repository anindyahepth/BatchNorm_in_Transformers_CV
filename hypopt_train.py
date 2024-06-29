# -*- coding: utf-8 -*-
"""HypOpt_train.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1GBbQ0_x8UgRyw3xnhBZb9Bik0FXQiSIo
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms as transforms_v2
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets as dsets

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.utils.notebook.plotting import render
from ax.utils.tutorials.cnn_utils import train, evaluate

from google.colab import files

from model.vitbnv1a import ViTBN


# GET THE DATASETS

def get_datasets() :
  data_transform = transforms.Compose([
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees = 0, translate = (0.2, 0.2)),
    transforms_v2.RandomZoomOut(0,(2.0, 2.0), p=0.2),
    transforms.Resize(28),
    transforms.ToTensor()
    ])

  #Load training data

  train_dataset = dsets.MNIST(root='./data', train=True, download=True, transform= data_transform)

  #Load validation data
  validation_dataset = dsets.MNIST(root='./data', train=False, download=True, transform= data_transform)

  return train_dataset, validation_dataset

# GET THE DATALOADERS

def get_dataloader(train_dataset, validation_dataset, batch_size):

  train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

  return train_loader, validation_loader


# DEFINE THE MODEL

def get_model(parameterization):
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

  #model.load_state_dict(torch.load("model100epoch_mnist.pth"))

  return model



# FUNCTION THAT RETURNS A TRAINED MODEL

def train_model(model, train_loader, parameters):

  # Define loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(),
                        lr=parameters.get("lr", 0.001), # 0.001 is default
  )
  n_epochs = parameters.get("n_epochs", 3)

  #training block

  for epoch in range(n_epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            model.train()
            z = model(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()

  return model


# Accuracy function

def eval_acc(model, data_loader):
    N_test = len(data_loader)
    for x_test,y_test in data_loader:
        model.eval()
        z = model(x_test)
        _, yhat = torch.max(z, 1)
        correct += (yhat == y_test).sum().item()
    accuracy = correct / N_test
    return accuarcy


# A Train_Evaluate Function that the Bayesian Opt calls on every run

def train_evaluate(train_dataset, valiadation_dataset, parameterization):

  # train_loader

  train_loader = torch.utils.data.DataLoader(train_datasetset,
                                batch_size=parameterization.get("batchsize", 100),
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True)

  validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                  batch_size= 500,
                                                  shuffle=True)


  # Get the untrained model

  model = get_model()

  # Get the trained model

  trained_model = train_model(model, train_loader, parameterization)

  # return the accuracy of the model

  return eval_acc(model = trained_model, data_loader=validation_loader)


# Now Optimize - this is where the ax service enters

best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-6, 0.4], "log_scale": True},
        {"name": "batchsize", "type": "range", "bounds": [20, 120]},
        #{"name": "momentum", "type": "range", "bounds": [0.0, 1.0]},
        #{"name": "max_epoch", "type": "range", "bounds": [1, 30]},
        #{"name": "stepsize", "type": "range", "bounds": [20, 40]},
    ],

    evaluation_function=train_evaluate,
    objective_name='accuracy',
)
print(best_parameters)
means, covariances = values
print(means)
print(covariances)



