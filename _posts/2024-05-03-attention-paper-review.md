---
date: 2024-05-03 14:45:55
layout: post
title: "Attention Is All You Need"
subtitle: "Understanding the foundational paper of Transformer architectures"
description: "Summarising and reviewing the transformer architecture and attention mechanism"
image: https://iili.io/JUWZiib.webp
optimized_image: https://iili.io/JUWZiib.webp
category: ml
tags: machine-learning nlp llm transformer attention
author: Govind Virdee
paginate: false
---

Natural language processing, large language models, transformers - we've all heard about them by now. In fact, I could have used ChatGPT or some other flavour of LLM to write this, but I'll try doing it the old fashioned way (though I may have used it to make the artwork for the cover). 

I don't know about you, but I sometimes forget how powerful it can be to do away with old, established ways of doing things in favour of radical new techniques. One such example is the foundation of the Transformer architecture - the first instance of which was written in the seminal paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). Previous to this work, language modeling and machine translation were dominated by architectures such as Recurrent Neural Networks, LSTM, and various encoder-decoder architectures. 

In a bold move, the authors put recurrence in the bin and relied on a mechanism of 'attention' to quite dramatically improve on the performance of previous models. In this article, we'll go through and understand the paper, exploring the wider context where necessary, and I'll focus on intuitively understanding and building the architecture brick-by-brick. I won't go through the actual training and results, as these are quite concise and direct in the paper itself (spoiler: it performed excellently). Feel free to skip any mathematical parts, though there aren't a huge amount anyway. 

I'll try my best to limit the number of Optimus Prime puns. 

## Contents

1. [Background](#background)
2. [Model Architecture](#model-architecture)
    - [Encoder-Decoder Structures](#encoder-decoder-structures)
    - [Back to Transformers](#back-to-transformers)
    - [Attention](#attention)
        - [Scaled Dot-Product Attention](#scaled-dot-product-attention)
        - [Multi-Head Attention](#multi-head-attention)
3. [Position-wise Feed-Forward Networks](#position-wise-feed-forward-networks)
4. [Embeddings and Softmax](#embeddings-and-softmax)
5. [Positional Encoding](#positional-encoding)
6. [Self Attention](#self-attention)


## Background 

The authors explain how many of the previous architectures struggled to model long-term dependencies in sequences; for example, how the start of a very long sentence relates to the end of a very long sentence. In some cases, the number of operations grows linearly, and for some, logarithmically. However, with the new architecture: 

> "In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution" 

We'll understand why the effective resolution is reduced later - but for now, we can already see the computational benefit. In essence, the number of operations required to compute dependencies between any two positions in the input does not depend on the distance between those positions. 

It achieves this through 'self-attention', which is succintly defined in the paper: 

> "Self-attention [...] is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence" 

We can think of an analogy for this. 

Imagine self-attention in a transformer model as a group of squirrels deciding where to stash their nuts for the winter. Each squirrel considers all the possible hiding spots they know of, but they want to choose the best ones based on their buddies' opinions. They peek at where their friends are hiding their nuts, decide whose stash strategy they trust the most, and then hide their nuts in a similar way, all while constantly chattering about who's the cleverest nut-hider. This whole process helps each squirrel find the optimal spot for its own nuts, just like how self-attention helps a model decide which parts of the data to focus on. 

An ideal analogy, if the squirrels were like matrices, and the chattering were dot-products. 

## Model Architecture 


![Not as good looking as Optimus Prime, in my opinion.](https://iili.io/JUXFGBp.png)

This looks complicated, at first glance. To help us understand it, let's first understand what encoder-decoder structures are. 

### Encoder-Decoder Structures 

Encoder-decoder structures are key architectural components in machine learning, particularly useful for tasks like translation and speech recognition.

The encoder component ingests input data, compressing it into a dense, feature-rich context or hidden state. It captures the essential aspects of the information, preparing it for further processing.

Operating on the context provided by the encoder, the decoder then reconstructs or translates this condensed information into an understandable output. Its job is to expand the compressed data back into a useful form.

The encoder-decoder framework is effective due to its division of labor. The encoder focuses solely on understanding and condensing the input, while the decoder specializes in reconstructing this data into the desired output. This separation enhances efficiency and accuracy, enabling each part to optimize its specific task without compromise.

In short, encoder-decoder structures are like a sophisticated system where one part distills information, and the other presents it, ensuring thorough understanding and precise communication.

Let me give you an unnecessary, tortured analogy. Let's go back to the squirrels. Imagine a squirrel duo handling their nut stash for winter: one squirrel, the encoder, carefully selects and compresses nuts into a tiny space, memorizing the best spots. The other squirrel, the decoder, later retrieves these nuts, using the encoder’s memory to expand them back into their mealtime stash. This teamwork ensures they efficiently manage their resources through the seasons. Is it a perfect analogy? No. Does it provide a good sense of what an encoder-decoder does? Maybe. 

### Back to transformers 

The transformer model follows an encoder-decoder structure, as you can see in the left and right sides of figure 1 above. The left side is the encoder, taking the input, and the right side is the decoder. 

The architecture looks like this (don't worry if the terms don't make sense, we'll go into them all in detail): 

**Encoder**: 

Layers: The encoder consists of 6 identical layers, with dimensions $d_{model} = 512$.

Sub-layers:

- Multi-head Self-Attention: Allows the model to focus on different positions of the input sequence simultaneously.
- Position-wise Feed-Forward Networks: These are fully connected layers applied to each position separately and identically.
Each sub-layer's output is added to its input (residual connection) and then normalized. This helps in training deep networks by mitigating the vanishing gradient problem.

**Decoder**:

Layers: Similar to the encoder, the decoder is made up of 6 identical layers.

Sub-layers:
- Masked Multi-head Self-Attention: Prevents future positions from being accessed, ensuring that the predictions are causal (a token can only attend to earlier tokens).
- Multi-head Attention over the Encoder's Output: Helps the decoder focus on relevant parts of the input sequence, facilitated by the encoder's output.
Position-wise Feed-Forward Networks: Same as those in the encoder.

The outputs are offset by one position, and self-attention in the decoder is masked to ensure each prediction can depend only on known outputs at previous positions.

We'll revisit some of these concepts soon, but this is just to paint the overall picture of the final result. Let's try to understand how these pieces fit together - starting with attention. 

## Attention 

We've had enough squirrel-based analogies describing things. Let's get more technical. 

Straight from the paper: 

> An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

What does this mean? Let's imagine the attention function is a spotlight that focuses on the most relevant pieces of information based on a question you ask, which we call the "query." Each piece of information, called a "value", has a label or tag known as a "key" that helps identify it. The function calculates how well each key matches your query, scoring them accordingly. These scores are called "weights". The final answer, or "output", is a blend of all the values, but with more emphasis on those that have higher scores. 

### Scaled Dot-Product Attention 

Now we're going to look at the type of attention used by the authors. They use "Scaled Dot-Product Attention". Why? The short answer is 'because it works, and is more efficient than the other types of attention'. Let's try and understand where it comes from. 

#### Step 1: Calculate the Dot Products of the Query with All Keys
First, the queries ($Q$) and keys ($K$) are matrices where each row represents a query or a key vector. The attention mechanism begins by calculating the dot product for each query with all keys. This dot product measures the similarity between a query and each of the keys.
 
<p align="center">
$\text{Attention Scores} = QK^T$
</p>

#### Step 2: Scale the Dot Products
Next, the raw scores are scaled down by dividing them by the square root of the dimension of the key vectors ($d_k$). 

<p align="center">
$\text{Scaled Attention Scores} = \frac{QK^T}{\sqrt{d_k}}$
</p>

This scaling is a crucial step because it helps in stabilizing the gradients during training, especially when the dimensions of the key vectors are large. (Side note - this is what makes this better than *additive* dot-product attention at large scales, which is another type of attention)
​
 
#### Step 3: Apply the Softmax Function
After scaling, a softmax function is applied to each row of the scaled scores. Softmax converts the scores into probabilities that sum to 1. This step emphasizes the keys that have higher scores, effectively determining which values (V) are most relevant to the queries.

<p align="center">
$\text{Attention Weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$
</p>

#### Step 4: Multiply by the Values
Finally, the attention weights (which are now probabilities) are used to compute a weighted sum of the value vectors (V). This final step blends the values based on how relevant each value is to the corresponding query, producing the output of the attention mechanism.

<p align="center">
$\text{Output} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$
</p>

Which is the final formula for the attention: 

<p align="center">
$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$
</p>

Let's summarise the intuition behind these steps:

**Dot Products**: Think of the dot products as a way to measure how much each key is related to a query.

**Scaling**: Scaling prevents numbers from getting too large, which can be problematic for numerical stability when computing softmax.

**Softmax**: By converting scores to probabilities, softmax helps focus the attention on the most relevant keys.

**Weighted Sum**: The final step blends the values based on these probabilities, focusing the output on the most relevant information for each query.

So this essentially represents a good, solid, efficient mathematical representation of how to implement attention, and it's worth understanding to some degree. 

![](https://iili.io/JgiVgWb.png) 

We can see all these steps in the figure above (Fig 2 in the paper). 

Now that we have some intuition behind this type of attention, let's see how to make it even more powerful. Wouldn't it be nice if we could add ANOTHER dimension to this style of attention? We're linking keys, values and queries together based on a particular 'learning', essentially a particular training run (and application of the usual backpropagation and optimisation). Let's add multiple learnings to a single model now. This is called Multi-Head Attention. 

### Multi-Head Attention 

Multi-head attention allows the model to capture different features in the data - each head can potentially focus on different features of the input data. It also allows us to process information in parallel, which can speed up our training as well as enrich the learning capabilities of the transformer. 

Now I'd like you to imagine multi-headed squirrels. Ok, let's not, but I still think the analogy might work if we try hard enough. 

Let's break down the formula for multi-head attention, step-by-step:

#### Step 1: Linear Projections
First, the inputs (queries $Q$, keys $K$, and values $V$) are linearly projected $h$ times with different, learnable linear transformations (i.e. trainings) for each head. These projections are denoted as $Q_i$, $K_i$ and $V_i$ for the $i^{th}$ head. 

<p align="center">
$Q_i = QW_i^Q, \quad K_i = KW_i^K, \quad V_i = VW_i^V$
</p>

where $W_i^Q$, $W_i^K$, and $W_i^V$ are parameter matrices for each head.

#### Step 2: Scaled Dot-Product Attention
Each head computes the scaled dot-product attention independently, as we went through before.

<p align="center">
$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i$
</p>

#### Step 3: Concatenation of Heads
After each head has computed its output, the outputs are concatenated, so we can bunch together all the outputs from these different heads.

<p align="center">
$\text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h)$
</p>

#### Step 4: Final Linear Projection
The concatenated result is then passed through a final linear transformation to combine the different learned representations into a single output vector. So, basically, we're taking all the separate head outputs and putting them into a single vector. 

<p align="center">
$\text{Output} = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W^O$
</p>
where $W^O$ is another learned weight matrix.

So putting it all together, the multi-head attention can be represented as:

<p align="center">
$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \text{head}_2, \dots, \text{head}_h) W^O$
</p>
with each head$_i$ computed as:
<p align="center">
$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right) V_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$
</p>

And boom, there we have multi-head attention, composed of separate heads all utilising the scaled dot-product attention we went through before. 

For completeness, in the paper they use $h = 8$ parallel attention layers (heads), with dimension = 64. 

Now, the authors go through how multi-head attention is used in the model. It turns out that there are 3 different applications. 


#### Encoder Self-Attention
In the first application, multi-head attention is used within each encoder layer. The queries, keys, and values all come from the same place — the output of the previous layer in the encoder (or initially, directly from the input embeddings). This type of attention allows each position in the encoder to attend to all positions in the previous layer of the encoder. This mechanism helps the model to capture information from ALL parts of the input sequence, integrating context from both nearby and distant elements within the input. 

#### Decoder Self-Attention
The second application is within the decoder. The difference here is that the attention mechanism is masked to prevent any given position from attending to subsequent positions in the output. This masking ensures that the predictions for a certain position can be made only using the known outputs at positions before it. Essentially, we're blinding the decoder from future positions in the sequence - it has to rely on being influenced by previous steps, which is important when building up coherent sequences (like sentences). It also maintains the 'auto-regressive' property, essentially meaning information is only passing in one way in the decoder. 

#### Encoder-Decoder Attention
The third application is in each layer of the decoder, allowing the decoder to attend to all positions in the output of the encoder. Here, the queries come from the previous decoder layer, and the keys and values come from the output of the encoder. This setup enables the decoder to focus on relevant parts of the input sentence, such as focusing on corresponding subject or object, which aids in producing coherent and contextually relevant outputs.

Ok, so we understand now how this multi-head attention fits in to the overall architecture. Let's revisit figure 1, and you'll notice you hopefully understand more now. 

![Still not as good looking as Optimus Prime.](https://iili.io/JUXFGBp.png)

You'll notice that there are some components we haven't gone through in the diagram - they are the feed forward networks, the input/output embeddings, positional encoding, and the softmax function at the end. 

Just like the authors do, let's go through these now, understanding them intuitively. 

## Position-wise Feed-Forward Networks 

The attention heads we've looked at allow for fantastic performance when looking globally at the input sequence, allowing for each position to be informed by the context of the whole sequence. If we want some more specific refinement, we can introduce some fairly vanilla feed-forward networks to complement the attention. Think of it like this: the position-wise feed-forward networks provide the necessary individual attention and complexity to each position in the sequence, complementing the broad, contextual processing done by the attention mechanisms. 

The position-wise feed-forward networks introduce non-linearity into the process. Each network typically consists of two linear transformations with a non-linear activation function (ReLU) in between. 

You might at this point be asking "What does position-wise mean?". That's a great question. Although the entire sequence (e.g., a sentence) is processed simultaneously thanks to the parallel architecture, the position-wise feed-forward networks apply their transformations to each position vector independently of others. This means that the operation on one position (one word) does not directly affect or change the operation on another. Additionally, each position goes through the same set of operations (the linear transformations + non-linear activation) but the input to these operations is unique for each position. For example, the word "king" in one position and "queen" in another position would each be transformed by the same neural network layers, but they start with different initial embeddings and possibly different contexts encoded from the attention layers. 

## Embeddings and Softmax 

Now let's look at embeddings. Embeddings are what allows us to to take the discrete, categorical input 'tokens' (manageable pieces of words, or subwords) and transform them into a format that neural networks can deal with - vectors of continuous values. These vectors represent the tokens in a high-dimensional space where similar tokens are positioned closer together. This representation captures semantic and syntactic properties of the tokens. 

It does so in three steps:  

**Tokenization**: the input data (e.g., text) is tokenized (split into manageable pieces as words or subwords). 

**Lookup**: Each token is then mapped to a vector using an embedding matrix, which is learned during training and is specific to the task and dataset. The vector for each token is retrieved from this matrix by index. 

**Positional Encoding**: We'll go through this next. 

Essentially, embedding is a protocol for turning words into vectors, where the vectors contain information about the similarity of those words to other words when compared to each other. Pretty useful stuff! 

Softmax is the other useful part of this - I won't go through the details of softmax here as it's a very standard function, but the point of it is to ensure that all of the attention weights sum to 1, allowing them to be interpreted as probabilities. This means we can effectively weight the values based on their relevance. 

## Positional Encoding 

So, we have embeddings, which allow us to convert and understand the words in a form that can be used in the architecture. How do we understand the position of each word, instead of treating a sentence as a random bag of words we're picking from? Since we're not using any recurrence or convolution in the transformer, we need another way to give this information.  

This is where positional encoding comes in. The main purpose of positional encodings is to provide the model with a way of understanding the order of tokens in the sequence. You'll notice if you look at the figure of the architecture that this is done right at the beginning, added to the embeddings - they have the same dimensions, so they can just be summed together. 

Let's look at the formulae used to assign the positional embeddings. 

For even indices $i$ (the $i^{th}$ dimension in the embedding): 

<p align="center">
$\text{PE}(pos, i) = \sin\left(\frac{pos}{10000^{i/d_{\text{model}}}}\right)$
</p>

and for odd indices $i$: 

<p align="center">
$\text{PE}(pos, i) = \cos\left(\frac{pos}{10000^{(i-1)/d_{\text{model}}}}\right)$
</p>

What the heck? Where did the sines and cosines come from? 

Although they might look confusing, there's actually a great reason for using trig functions like this here. 

First, let's remind ourselves of what we want - we want for each token (part of a word), a representation of this in a vector, which contains not only information about how similar it is to other words, but also *where* in the sentence it falls. 

Why are sine and cosine useful for this? Well, there's a few reasons: 

**Understanding Position**: Each position in a sequence needs a unique identifier. We use sine and cosine functions because they vary smoothly and continuously, which helps a neural network learn effectively.

**Periodic Functions**: We choose sine and cosine because they are periodic functions, meaning their values repeat in cycles. This cyclical property helps the model to learn and predict sequences where the relative positioning of elements can suggest periodicity or recurrence in data patterns.

**Modulating Frequency**: To ensure each position has a unique encoding and that similar positions can still be recognized by the model, we modulate the frequency of the sine and cosine waves. The modulation allows the encodings to capture not just the individual position but also how positions relate to each other.

So we end up with a bunch of different sine and cosines of different frequencies, which can be used to contain the information about the positions. 

Let's understand the components of the formulae above. 

Each dimension of the positional encoding vector uses a sine or cosine function, but with different wavelengths for each dimension.

Variable Definitions:

$pos$ is the position in the sequence.

$i$ is the dimension in the positional encoding vector.

$d_{model}$ is the total number of dimensions in the model (and in each positional encoding vector).

Frequency Adjustment:
We need the frequencies to vary across dimensions. To achieve this, we use the formula $10000^{i/d_{model}}$, which gives a wide range of wavelengths from $2\pi$ to $10000 \cdot 2\pi$ - essentially a large range to ensure we can possibly contain all the representations of positions for the words.

Let's visualise this a bit. Below, I've plotted the first 4 dimensions (of potentially hundreds) of an encoding using the above formulae. 

![](https://iili.io/Jgsa6oQ.png) 

Looking at this, what should we take away? Well, imagine we're encoding the information for the position of a word relative to others. We know that for example, the word 'blue' might come before the word 'sky', but it might also come before the word 'car'. In a sentence the grammar, syntax, and other aspects would probably be very similar, but there's a slight difference. Positionally, it would also be similar too. We'd like an efficient set of mathematical functions that play nicely with machine learning algorithms (from a computational perspective), and that can also give different (but relatively similar!) values for the position of a word used in two slightly different contexts. That's the plot above! Imagine 512 lines, all very close together, but based on the same underlying function. Plenty of points to choose from.  

Ok, now hopefully we have an understanding of the components of the architecture. Now let's see the authors' justification for why self-attention is a great thing. 

## Self Attention 

The authors mention three desiderata (things to be desired) when measuring the performance of the transformer in comparison to other architectures. These are capturing long-range dependencies, reducing computational complexity per layer, and the amount of computation that can be parallelised. Let's see what they say about these. 

Traditional architectures like recurrent neural networks (RNNs) and long short-term memory (LSTM) networks struggle to effectively model long-range dependencies in sequences. As the distance between elements in a sequence increases, these architectures often face challenges in capturing relationships between distant elements. This limitation results in either linear or logarithmic growth in computational complexity, making it difficult to process long sequences efficiently. The authors (rightly) say that self-attention mechanisms offer a solution to this problem by allowing the model to capture dependencies between any two positions in a sequence with a constant number of operations. By attending to different positions of the input sequence simultaneously, self-attention enables the model to integrate context from both nearby and distant elements efficiently - so, long-range dependencies are in the bag.

As for the second and third desired things, self-attention reduces computational complexity by enabling parallel processing of input sequences. Unlike traditional architectures where the computational cost increases with the distance between elements, self-attention mechanisms ensure that the number of operations remains constant regardless of the distance between positions - an immediate advantage in terms of computation. 

They also add the valid point that self-attention is relatively interpretable. The heads of the model learn tasks that are similar in nature to standard grammatical and syntatical features of language (pretty cool)! 

## I hope you paid attention! 

Well, that's it - we've gone through a fairly in-depth dive into an intuitive way of understanding the transformer architecture. Please do check out the [original paper](https://arxiv.org/abs/1706.03762), and be sure to keep an eye out for future posts, where I'll likely dive in to other architectures and features in more depth! 