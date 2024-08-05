
Batch Normalization is a powerful regularization method for Neural Networks which can 
substantially speed up the learning process in CNNs. We investigate the impact of Batch Normalization 
in the most basic Transformer-based model for image classification - the Vision Transformer (ViT). 
We consider two distinct models:

1. **ViTBNFFN** : This implement a batchnorm layer in the Feed Forward Netweork (FFN) component of the standard ViT.
2. **ViTBN** : This replaces all LayerNorms with BatchNorms.


The model ViTBN (and analogously ViTBNFFN) can be used in the following fashion:

```
from model.vitbn import ViTBN

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
by setting the hyperparameter `pool' to 'cls' or 'mean' respectively. One can also choose a learnable positional encoding vector or a 1d 
sinusoidal vector, indicated by setting 'pos_emb ' to 'learn' or 'pe1d' respectively.



For comparing the models using the MNIST dataset, we use the following architectures for the three models:
```
ViT(image_size = 28,
    patch_size = 7,
    num_classes = 10,
    channels =1,
    dim = 64,
    depth = 6,
    heads = 8,
    mlp_dim = 128,
    pool = 'cls',
    dropout = 0.0,
    emb_dropout = 0.0),

ViTBNFFN(image_size = 28,
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
    pos_emb ='learn'),

ViTBN(image_size = 28,
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
    pos_emb ='learn')

```
Note that ViT uses the learnable positional embedding if 'pool' = 'cls'. 
Training on the MNIST dataset of handwritten digits from scratch, we see that the models with BatchNorm are 
about 60% faster than the standard ViT in terms of the average inference time per epoch. 
The gain in speed for the average training time per epoch can be even higher. As an example, one may consider 
training and testing the models where the learning rate and batch size in each case is determined by a Bayesian optimization 
procedure. 
The following graphs compare 
the performances of the models where the learning rate and batch size in each case by a Bayesian optimization 
procedure. 

![image](https://github.com/user-attachments/assets/ea3ae9fa-bd91-44c8-a0a2-3807953c8a00)



![TrainDur](https://github.com/anindyahepth/BatchNorm_in_Transformers_CV/assets/129802283/d1a0a7fd-f6e1-4e64-8872-a1520a64460b)


![TestDur](https://github.com/anindyahepth/BatchNorm_in_Transformers_CV/assets/129802283/5d446ef1-11c7-446f-8020-9af584df01ac)

![TestAcc](https://github.com/anindyahepth/BatchNorm_in_Transformers_CV/assets/129802283/91bab246-f389-48ea-9713-793e47ff6f5b)


Finally, we train ViTBN on the MNIST data for 100 epochs and use the trained model to 
build a web app for recognizing handwritten digits. The webapp can be found 
here : https://anindyadey.pythonanywhere.com/
The app has an html-css-javascript frontend and a Flask backend. 





