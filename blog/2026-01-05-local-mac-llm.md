---
title: "Training Your Own LLM on a MacBook in 10 Minutes"
subtitle: "De-mystifying transformers by building a 1.5M parameter model from scratch"
date: 2026-01-05
author: "Sainath Krishnamurthy"
tags: ["AI", "Machine Learning", "MLX", "LLM", "Cursor", "Apple Silicon"]
---

# Training Your Own LLM on a MacBook in 10 Minutes

We have reached a point where the tools for building large language models are no longer exclusive to massive data centers. While the industry focuses on models with hundreds of billions of parameters, there is immense value in stripping away the complexity to see how these systems actually work.

My recent project, **LocalMacLLM**, was an attempt to do exactly that: build a fully functional, 1.5-million parameter GPT-style model that trains entirely on a MacBook Pro in under ten minutes.

### The Spark of an Idea

The project was inspired by the realization that you don't need a massive GPU cluster to understand the core mechanics of a transformer. Following a guide by Sean Goedecke ["Training a Model on a MacBook Pro (M1 Pro)"](https://www.seangoedecke.com/model-on-a-mbp/), I set out to build a pipeline that could take raw text and turn it into a generative model using Apple’s MLX framework.

The goal wasn't to compete with frontier models, but to create a "miniature" version—a model trained on the TinyStories dataset that could generate simple, coherent narratives for children. It’s a complete end-to-end implementation that covers everything from data preprocessing to real-time inference.

### Building with an AI Partner

Developing this project was as much an experiment in workflow as it was in machine learning. I used the **Cursor AI agent** to assist in the development, and the experience was revealing. 

Instead of manually writing every line of the transformer architecture or the data loading utilities, I worked alongside the agent to iterate on the design. It helped bridge the gap between my conceptual understanding of how a transformer should work and the specific implementation details required by the MLX library. This collaboration allowed me to focus on the high-level logic and the learning process itself, rather than getting bogged down in boilerplate code.

### The Technical Reality of a 1.5M Model

Despite its small size, the model follows the standard GPT architecture. It uses a 7-layer transformer with a 4-head attention mechanism and a context window of 256 tokens. 

One of the most interesting parts was the tokenization. We used SentencePiece to train a custom BPE tokenizer with a vocabulary of just 1,024 tokens. This small vocabulary is perfectly suited for the limited world of TinyStories, allowing the model to focus its limited capacity on learning the structure of language rather than memorizing a massive dictionary.

On an M1 Pro MacBook, the model reaches a respectable perplexity of around 9.6 after just a few minutes of training. It’s a reminder that efficiency and scale are two different levers, and for educational purposes, efficiency wins.

### What I’ve Learned About Small Models

This experiment changed how I think about AI models in several ways:

1.  **Complexity isn't always quality**: A tiny model, when trained on high-quality, specialized data like TinyStories, can produce remarkably coherent text.
2.  **The importance of the pipeline**: The most challenging parts of building an LLM aren't the model weights themselves, but the data packing, tokenization, and memory-mapped loading that keep the training efficient.
3.  **Local is viable**: Running training loops locally on a laptop is not just possible; it’s an incredibly fast way to iterate and learn.

### What’s Next

LocalMacLLM is a proof of concept for local, specialized intelligence. The next step is to explore how these small models can be fine-tuned for even more specific tasks or how they might interact with larger models in a multi-agent system.

The barrier to entry for understanding AI has never been lower. You just need a bit of curiosity and the machine that’s already on your desk.

---
*Sainath is a designer, researcher, and technologist exploring the intersection of human experience and technical implementation. You can follow more of his work at [OpusLABS](https://opuslabs.substack.com).*
