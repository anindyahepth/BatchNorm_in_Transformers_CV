
Batch Normalization is a powerful regularization method for Neural Networks which can 
substantially speed up the training process by allowing a higher learning rate and 
removing dropout. We investigate the impact of Batch Normalization in Transformer-based 
models tailored to Computer Vision tasks. As a first step, we implement a batchnorm layer 
in the Feed Forward Netweork (FFN) component of a Vision Transformer (ViT). We call this 
a ViTBN (Vision Transformer with BatchNorm).

Training on the FashionMNIST(FMNIST) and MNIST datasets from scratch, we see that ViTBN 
is about 3-4 times faster compared to the vanilla ViT per epoch both during training 
and validation.  





https://anindyadey.pythonanywhere.com/
