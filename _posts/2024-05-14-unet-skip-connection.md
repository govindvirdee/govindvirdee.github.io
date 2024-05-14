---
date: 2024-05-14 08:17:37
layout: post
title: "Skip your way to better image segmentation"
subtitle: Understanding the U-Net architecture and the role of skip connections 
description:
image: https://iili.io/JrGUtB2.webp
optimized_image: https://iili.io/JrGUtB2.webp
category: ML 
tags: machine-learning computer-vision skip-connections u-net
author: Govind Singh Virdee 
paginate: false
---

I'll be posting an article soon about a medical imaging segmentation project I'm working on, but I thought I'd do a primer first on U-Net architectures and skip connections, which I can link back to in that article and in doing so perhaps feel very accomplished with myself. 

This article will go through the broad strokes of the U-Net architecture, which is a type of convolutional neural network (CNN) that performs exceptionally well in medical image segmentation (finding things in medical images) thanks to the use of 'skip connections'. 

Let's get right into it! 

(The AI generated image above is what GenAI thinks a U-Net looks like, complete with gibberish labels, so there's that!)

## U-Net 

The U-Net architecture, since its [inception in 2015](https://arxiv.org/abs/1505.04597) by Olaf Ronneberger, Philipp Fischer, and Thomas Brox, has become a cornerstone in the field of image segmentation, especially in medical imaging. 

U-Net is widely known for its efficiency in image segmentation tasks, particularly where data is sparse and high precision is critical. Its design is quite intuitive, resembling the shape of the letter "U", which is split into two main pathways: the contraction (downsampling) and the expansion (upsampling) paths. We'll break this down now by going through both. I'll assume basic knowledge of neural nets as we go through. 

Here's a U-net in all its glory (taken from the paper linked above). 

![](https://iili.io/J6YeoR2.png)


### Contraction Path

The contraction path is a sequence of layers designed to process and downsample the image:

1. **Convolution Layers:** These layers apply filters to the image to extract features such as edges, textures, etc, as you'd find in a vanilla CNN, using a ReLU activation function. 
2. **Max Pooling:** This reduces the dimensionality of each feature map while retaining the most important information.

Each step in this path reduces the spatial dimensions of the image but increases the depth, enhancing the network’s ability to interpret complex features at a reduced computational cost. If you're familiar with CNNs, you can think of the contraction path as essentially a CNN in itself. 

### Expansion Path

Conversely, the expansion path aims to project the lower-resolution encoder feature maps back to the higher resolution space to pinpoint the exact boundaries of the object of interest:

1. **Up-Convolution:** These layers increase the dimensions of the feature maps.
2. **Concatenation with Skip Connection:** Feature maps from the contraction path are concatenated with the upsampled features to preserve high-resolution details.
3. **Convolution Layers:** Further refine the features to produce the final image segmentation.

So we have essentially the reverse here - once the contraction has occured, we then upscale the resulting features that we've learned from the contraction to increase the number of dimensions and allow the network to segment more effectively. 

Let's focus on point 2 above - what are these skip connections, and why do they help preserve high resolution details? 

### The Role of Skip Connections

Skip connections are essentially shortcuts that bypass one or more layers in the network and pass earlier information directly to later layers, essentially "skipping" over intermediate processing. 

Why do we bother with this? 

- **Feature Reusability:** They allow the network to reuse features from earlier layers, helping to restore information lost during downsampling.
- **Improved Gradient Flow:** They provide alternative pathways for the gradient during backpropagation, which can help address the vanishing gradient problem in deep networks.
- **Better Localization:** By combining low-level, detailed features with high-level, abstract features, skip connections help the network better localize and delineate complex objects.

So we end up getting the best of both worlds, here - the generalisability from the downsampling to identify the key, most powerful features for the segmentation task at hand, plus the fine details that would otherwise have been lost in that downsampling which are preserved through these skip connections. 

### A bit of maths 

Consider a simple example where a neural network layer's output $( Y )$ is given by:

<p align="center">
$ Y = ReLU(WX + b) $
</p>

where $W$ and $b$ are weights and biases, $X$ is the input, and ReLU is the activation function.

In a network with a skip connection, the output of the ReLU activation would be modified to:

<p align="center">
$Y = ReLU(WX + b + X)$
</p>

This modification, where $X$ (input) is added directly to the output of the layer, represents a basic skip connection that helps preserve the input through layers of transformations. 

You can see it represented on the U-Net diagram at the start of the article - it's the grey arrows going from left to right, labelled "copy and crop". 

### Practical Example

Here’s a simple Python example using TensorFlow to implement a basic U-Net with skip connections:

```python
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate

def unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Encoder
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)

    # Decoder
    up1 = UpSampling2D(size=(2, 2))(conv3)
    merge1 = concatenate([conv2, up1], axis=3) # Here's the skip connection! We're concatenating the corresponding layer in the encoder (contraction) to the current layer, preserving the details that would have been lost. 
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    conv4 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)

    up2 = UpSampling2D(size=(2, 2))(conv4)
    merge2 = concatenate([conv1, up2], axis=3)
    conv5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge2)
    conv5 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    # Output layer
    output = Conv2D(1, 1, activation='sigmoid')(conv5)

    model = Model(inputs=inputs, outputs=output)

    return model

# Create the U-Net model
model = unet_model()
model.summary()

``` 

And here's a little run-through: 

**Input**: The input layer takes an image of size 256x256 with 1 channel (for grayscale images).

**Encoder**: This consists of two blocks of two convolutional layers each. After each block, a max pooling operation reduces the spatial dimensions by half.

**Bottleneck**: This is the lowest level of the network with fewer spatial dimensions and higher feature dimensions.

**Decoder**: This includes upsampling layers that increase the resolution of the output. After each upsampling, a skip connection is made by concatenating feature maps from the corresponding encoder block. This helps in recovering spatial information lost during downsampling.

**Output**: A final convolutional layer reduces the number of output channels to 1, which represents the segmented output.

Fairly simple, yet very powerful! 

## Summary 

U-Net’s architecture and its use of skip connections exemplify how strategic network design can lead to significant improvements in model efficacy, especially in tasks requiring precise localization like medical image segmentation. Its principles can also extend to various other domains such as satellite imagery, autonomous vehicle navigation, and more, proving its versatility and robustness - pretty cool.

So next time you use a U-Net, or a ResNet, or any other analogous net - just remember all the fine details SHOULD be skipped. 


