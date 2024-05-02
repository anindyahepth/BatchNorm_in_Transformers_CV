
Batch Normalization is a powerful regularization method for Neural Networks which can 
substantially speed up the training process by allowing a higher learning rate and 
removing dropout. We investigate the impact of Batch Normalization in Transformer-based 
models tailored to Computer Vision tasks. As a first step, we implement a batchnorm layer 
in the Feed Forward Netweork (FFN) component of the standard Vision Transformer (ViT). 
We call this a ViTBN (Vision Transformer with BatchNorm).

Training on the FashionMNIST(FMNIST) and MNIST datasets from scratch, we see that ViTBN 
is about 3 times faster compared to the standard ViT per epoch both during training phase
and about 5 times faster during the testing phase. In addition, the former reaches the 
highest accuracy after training for a far fewer number of epochs. The relevant graphs are 
attached and explained below. 

Finally, we train ViTBN on the MNIST data for 100 epochs and use the trained model to 
build a Flask-based app for recognizing handwritten digits. The webapp can be found 
here : https://anindyadey.pythonanywhere.com/


The model can be loaded and used as follows. 

```
from model.vitbnv1 import ViTBN

model = ViTBN(
                image_size = 28,
                patch_size = 7,
                num_classes = 26,
                channels =1,
                dim = 64,
                depth = 6,
                heads = 8,
                dim_head = 64,
                mlp_dim = 128,
                pool = 'cls' or 'mean',
                dropout = 0.0,
                emb_dropout = 0.0,
                pos_emb ='learn' or 'pe1d'
    )
```
One can choose either cls tokens or a global pooling to implement the final classification using the MLP head, indicated 
by setting `pool' to 'cls' or 'mean' respectively. One can also choose a learnable positional encoding vector or a 1d 
sinusoidal vector, indicated by setting 'pos_emb ' to 'learn' or 'pe1d' respectively.

![TrainDur](https://github.com/anindyahepth/BatchNorm_in_Transformers_CV/assets/129802283/d1a0a7fd-f6e1-4e64-8872-a1520a64460b)


![TestDur](https://github.com/anindyahepth/BatchNorm_in_Transformers_CV/assets/129802283/5d446ef1-11c7-446f-8020-9af584df01ac)

![TestAcc](https://github.com/anindyahepth/BatchNorm_in_Transformers_CV/assets/129802283/91bab246-f389-48ea-9713-793e47ff6f5b)








