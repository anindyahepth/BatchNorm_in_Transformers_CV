import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
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

from model.vitbnffn import ViTBNFFN
from model.vit_org import ViT
from emnist_digit_preprocessing import download_emnist
from emnist_digit_preprocessing import MNISTCustomDataset


# GET THE DATASETS

def get_datasets_emnist() :
  dir_root = '/content/'
  file_dict={
    'train_images':'emnist-letters-train-images-idx3-ubyte.gz',
    'train_labels':'emnist-letters-train-labels-idx1-ubyte.gz',
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
  train_dataset = dsets.FashionMNIST('~/.pytorch/F_MNIST_data', train=True, download=True,transform= data_transform)

#Load the validation dataset

  validation_dataset = dsets.FashionMNIST('~/.pytorch/F_MNIST_data', train=False, download=True, transform= data_transform)

  return train_dataset, validation_dataset
	


# GET THE DATALOADERS

# def get_dataloader(train_dataset, validation_dataset, batch_size):

#   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#   validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)

#   return train_loader, validation_loader


# DEFINE THE MODEL

# def get_model(parameterization):
#   model = ViTBNFFN(
#                 image_size = 28,
#                 patch_size = 7,
#                 num_classes = 10,
#                 channels =1,
#                 dim = 64,
#                 depth = 6,
#                 heads = 8,
#                 mlp_dim = 128,
#                 pool = 'cls',
#                 dropout = 0.0,
#                 emb_dropout = 0.0,
#                 pos_emb ='learn'
#     )
    
#   depth = parameterization.get("depth", 2)

#   #model.load_state_dict(torch.load("model100epoch_mnist.pth"))

#   return model



# FUNCTION THAT RETURNS A TRAINED MODEL

def train_model(model, train_loader, parameters):

  # Define loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(),
                        lr=parameters.get("lr", 0.001), # 0.001 is default
                        weight_decay=parameters.get("lambda", 0.), #default is 0 weight decay
  )
  scheduler = optim.lr_scheduler.StepLR(
      optimizer,
      step_size=int(parameters.get("step_size", 1)),
      gamma=parameters.get("gamma", 1.0),  # default is no learning rate decay
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
        scheduler.step()

  return model


# Accuracy function

def eval_acc(model, data_loader, validation_dataset):
    N_test = len(validation_dataset)
    correct = 0
    accuracy = 0
    for x_test,y_test in data_loader:
        model.eval()
        z = model(x_test)
        _, yhat = torch.max(z, 1)
        correct += (yhat == y_test).sum().item()
    accuracy = correct / N_test
    return accuracy


# A Train_Evaluate Function that the Bayesian Opt calls on every run

def train_evaluate(parameterization):

  # train_loader

  train_loader = torch.utils.data.DataLoader(train_dataset,
                                batch_size=parameterization.get("batchsize", 100),
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True)

  validation_loader = torch.utils.data.DataLoader(validation_dataset,
                                                  batch_size= 5000,
                                                  shuffle=True)


  # Get the untrained model

  model = ViT(
                image_size = 28,
                patch_size = 7,
                num_classes = 10,
                channels =1,
                dim = 64,
                depth = parameterization.get("depth", 2),
                heads = 8,
                mlp_dim = 128,
                pool = 'cls',
                dropout = 0.0,
                emb_dropout = 0.0,
                #pos_emb ='learn'
    )

  # Get the trained model

  trained_model = train_model(model, train_loader, parameterization)

  # return the accuracy of the model

  return eval_acc(model = trained_model, data_loader=validation_loader, validation_dataset = validation_dataset)


# Now Optimize - this is where the ax service enters


if __name__ == "__main__":

    train_dataset, validation_dataset = get_datasets_mnist()
    
best_parameters, values, experiment, model = optimize(
    parameters=[
        {"name": "lr", "type": "range", "bounds": [1e-3, 1e-1], "log_scale": True},
        #{"name": "batchsize", "type": "range", "bounds": [20, 120]},
        {"name": "gamma", "type": "range", "bounds": [0.95, 1.0]},
        {"name": "stepsize", "type": "range", "bounds": [1, 3]},
        {"name": "depth", "type": "range", "bounds": [1, 10]},
        #{"name": "lambda", "type": "range", "bounds": [1e-4, 1e-1], "log_scale": True},
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

best_objectives = np.array([[trial.objective_mean*100 for trial in experiment.trials.values()]])

best_objective_plot = optimization_trace_single_method(
    y=np.maximum.accumulate(best_objectives, axis=1),
    title="Model performance vs. # of iterations",
    ylabel="Classification Accuracy, %",
)
render(best_objective_plot)

#render(plot_contour(model=model, param_x='batchsize', param_y='lr', metric_name='accuracy'))


