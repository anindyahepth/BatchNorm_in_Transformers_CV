
Batch Normalization is a powerful regularization method that can 
substantially speed up the learning process in deep neural networks. On the other hand, LayerNorm is used as 
the default normalization technique in Transformers. We investigate the impact of Batch Normalization 
in the most basic Transformer-based model for image classification - the Vision Transformer (ViT). 

We consider two distinct models:

1. **ViTBNFFN** : This implements a batchnorm layer in the Feed Forward Netweork (FFN) component of the standard ViT.
2. **ViTBN** : This replaces all LayerNorms with BatchNorms.


The model ViTBNFFN (and analogously ViTBN) can be used in the following fashion:

```
from model.vitbnffn import ViTBNFFN

model = ViTBNFFN(
                image_size = 28,
                patch_size = 7,
                num_classes = 10,
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
    depth = 1,
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
    depth = 1,
    heads = 8,
    mlp_dim = 128,
    pool = 'cls',
    dropout = 0.0,
    emb_dropout = 0.0,
    pos_emb ='learn')

```
Note that ViT uses the learnable positional embedding by default if 'pool' = 'cls'. 

Training on the MNIST dataset of handwritten digits from scratch, we see that the models with BatchNorm are 
about 60% faster than the standard ViT in terms of the average inference time per epoch. 
The gain in speed for the average training time per epoch can be even higher. 

As an example, consider training and testing the models with the learning rate and the batch size in 
each case being determined by a Bayesian optimization procedure. The following graphs compare 
the performances of the optimized models on four metrics as functions of epochs - the training time (in seconds), 
the testing time (in seconds), the training loss and test accuracy. 

![image](https://github.com/user-attachments/assets/ea3ae9fa-bd91-44c8-a0a2-3807953c8a00)

For the full story, check out the article : https://medium.com/@anindya.hepth/speeding-up-the-vision-transformer-with-batch-normalization-d37f13f20ae7 

Finally, we train the ViTBNFFN model on the MNIST data for 100 epochs and use the trained model to 
build a web app with a Flask back-end for recognizing handwritten digits. The webapp can be found 
here : https://anindyadey.pythonanywhere.com/





