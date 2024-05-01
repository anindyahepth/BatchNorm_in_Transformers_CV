
Batch Normalization is a powerful regularization method for Neural Networks which can 
substantially speed up the training process by allowing a higher learning rate and 
removing dropout. We investigate the impact of Batch Normalization in Transformer-based 
models tailored to Computer Vision tasks. As a first step, we implement a batchnorm layer 
in the Feed Forward Netweork (FFN) component of a Vision Transformer (ViT). We call this 
a ViTBN (Vision Transformer with BatchNorm).

Training on the FashionMNIST(FMNIST) and MNIST datasets from scratch, we see that ViTBN 
is about 3-4 times faster compared to the vanilla ViT per epoch both during training 
and validation. In addition, the former reaches the highest accuracy after training for 
a far fewer number of epochs. 

Finally, we train ViTBN on the MNIST data for 100 epochs and use the trained model to 
build a Flask-based app for recognizing handwritten digits. The webapp can be found 
here : https://anindyadey.pythonanywhere.com/


The model 

from model.vitbnv1 import ViTBN

model = ViTBN(
                image_size = 28,
                patch_size = 7,
                num_classes = 26,
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







