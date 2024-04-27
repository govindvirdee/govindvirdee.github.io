---
date: 2024-04-27 11:50:46
layout: post
title: "Understanding the Tversky loss function"
subtitle: "Getting to grips with improving segmentation in medical imaging"
description: "A dive into the Tversky loss function to deal with imbalanced classes in imaging datasets"
image: "https://iili.io/JUGxkXf.png"
optimized_image: "https://iili.io/JUGxkXf.png" 
category: ml
tags: computer-vision medical-imaging keras tensorflow loss-functions tversky 
author: Govind Virdee
paginate: false
---

Lately, I've been working on an image segmentation project for identifying brain tumours in MRI scans. For context, brain tumours are bad. Fortunately, they're not incredibly prevalent. While this is good for humanities' sake, it does mean that datasets containing MRI images can, at times, be rather imbalanced - we have fewer examples of MRI scans with brain tumours than scans without. In this context, we'd also very much like to reduce the number of false negatives. 

When we look in our loss function toolbox, we tend to find that many traditional loss functions such as mean-squared-error or cross-entropy can fall short when it comes to imbalanced class distributions. This is where a custom loss function might be more useful! 

### Quick reminder: Loss Functions

Before we dive into our custom loss function, let's quickly remind ourselves of what a loss function is. In machine learning, a loss function is a method used to estimate the difference between the actual outputs of the model and the expected outputs. It's the way we guide the training process — essentially, the model learns by attempting to minimize this loss over time.

## The Tversky Index

The Tversky index is a way to measure how similar two sets of items are, but it does so in a way that allows you to give more importance to certain types of mistakes over others. This measure builds on simpler similarity measures (like the Dice coefficient and the Jaccard index() by adding weights that affect how much you care about false positives (mistakes where something is incorrectly included) and false negatives (mistakes where something is incorrectly left out).

Amos Tversky introduced this index back in 1977 primarily for studying how people think and make decisions. More recently, it's being used in artificial intelligence, particularly in predictive modeling, where it helps tailor the handling of errors in ways that are most useful for specific applications.

## The Formula Behind Tversky Loss

The mathematical formulation of the Tversky index is as follows:

Tversky Index $(TI) = \frac{TP}{TP + \alpha FP + \beta FN}$

Where:
- TP is true positives,
- FP is false positives,
- FN is false negatives,
- $\alpha$ and $\beta$ are parameters that control the weighting of false positives and false negatives, respectively.

The beauty of the Tversky index lies in its flexibility; by adjusting $\alpha$ and $\beta$, we can make the model more sensitive to false negatives or false positives, tailoring our loss function to the specific needs of our task.

## Tversky Loss

In practice, the Tversky loss function is implemented by subtracting the Tversky index from 1:


Tversky Loss $= 1 - TI$


This transformation turns it into a loss value that can be minimized. During the training of neural networks, particularly in segmentation tasks, minimizing the Tversky loss directly correlates to maximizing the Tversky index, pushing the model towards higher precision and better handling of class imbalance.

## Real-World Applications

The practical applications of the Tversky loss function are vast but most notable in medical imaging, where the cost of missing a condition (false negative) is much higher than incorrectly identifying one (false positive). For instance, in tumor detection, missing a tumor could be life-threatening, whereas a false alarm might simply lead to further testing. By tweaking $\alpha$ and $\beta$, we can adjust the sensitivity of our models to ensure that they catch as many true cases as possible without overwhelming the system with false alarms.

## Implementing Tversky Loss in Python

For those interested in how to implement this in a practical setting, here's a brief code snippet using TensorFlow/Keras:

```python
import tensorflow as tf

def tversky_loss(y_true, y_pred):
    alpha = 0.5  # Example value; adjust as needed
    beta = 0.5   # Example value; adjust as needed
    ones = tf.ones(tf.shape(y_true))
    p0 = y_pred      # Probability that voxels are class i
    p1 = ones - y_pred  # Probability that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = tf.reduce_sum(p0 * g0, (0,1,2))
    den = num + alpha * tf.reduce_sum(p0 * g1, (0,1,2)) + beta * tf.reduce_sum(p1 * g0, (0,1,2))

    T = tf.reduce_sum(num / den)  # Sum over classes

    return 1 - T  # Return Tversky loss

```

Stay tuned - I'll be doing another blog post soon where you can see this in action! 