---
date: 2024-04-27 18:14:19
layout: post
title: "ML Journal Club: Attention Is All You Need"
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

In a bold move, the authors put recurrence in the bin and relied on a mechanism of 'attention' to quite dramatically improve on the performance of previous models. In this article, we'll go through and understand the paper, exploring the wider context where necessary. I'll try my best to limit the number of Optimus Prime puns. 

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

### Tangent: Encoder-Decoder Structures 

Encoder-decoder structures are key architectural components in machine learning, particularly useful for tasks like translation and speech recognition.

The encoder component ingests input data, compressing it into a dense, feature-rich context or hidden state. It captures the essential aspects of the information, preparing it for further processing.

Operating on the context provided by the encoder, the decoder then reconstructs or translates this condensed information into an understandable output. Its job is to expand the compressed data back into a useful form.

The encoder-decoder framework is effective due to its division of labor. The encoder focuses solely on understanding and condensing the input, while the decoder specializes in reconstructing this data into the desired output. This separation enhances efficiency and accuracy, enabling each part to optimize its specific task without compromise.

In short, encoder-decoder structures are like a sophisticated system where one part distills information, and the other presents it, ensuring thorough understanding and precise communication.

Let me give you an unnecessary, tortured analogy. Let's go back to the squirrels. Imagine a squirrel duo handling their nut stash for winter: one squirrel, the encoder, carefully selects and compresses nuts into a tiny space, memorizing the best spots. The other squirrel, the decoder, later retrieves these nuts, using the encoder’s memory to expand them back into their mealtime stash. This teamwork ensures they efficiently manage their resources through the seasons. Is it a perfect analogy? No. Does it provide a good sense of what an encoder-decoder does? Maybe. 

### Back to transformers 

The transformer model follows an encoder-decoder structure, as you can see in the left and right sides of figure 1 above. The left side is the encoder, taking the input, and the right side is the decoder. 








