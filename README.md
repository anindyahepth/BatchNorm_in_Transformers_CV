
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


Training on the MNIST dataset of handwritten digits from scratch, we make the following observations:

1. For a reasonable choice of hyperparameters, ViTBNFFN does converge faster than ViT, provided the transformer depth (i.e number of layers in the encoder) is sufficiently large.
2. As one increases the learning rate, ViTBNFFN turns out to be more stable than ViT, especially at larger depths. 

As an illustrative example, consider the accuracy curves of the two models trained with learning rate lr=0.003 and batch size 100 for depths d=4,5,6,7.

![image](https://github.com/user-attachments/assets/3eff49ee-9ac5-4c04-b66c-89eda38a9947)

For d=6 and above, the model with BatchNorm achieves a much higher accuracy than the standard ViT. 
 

Finally, we train the ViTBNFFN model on the MNIST data for 100 epochs and use the trained model to 
build a web app with a Flask back-end for recognizing handwritten digits. The webapp can be found 
here : https://anindyadey.pythonanywhere.com/





