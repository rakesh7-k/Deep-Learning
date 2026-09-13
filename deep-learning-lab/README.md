# Deep Learning From Neurons to Generative Models

This repository documents a progression through core Deep Learning ideas, beginning with the earliest artificial neuron and moving through binary classification, neural networks, convolutional models, sequence learning, and generative modeling. The goal is to make the ideas easy to follow while keeping the structure professional and portfolio-ready.

## Roadmap

Artificial Neuron
↓
Logistic Regression
↓
Neural Network
↓
CNN
↓
Transfer Learning
↓
Autoencoder
↓
RNN
↓
LSTM
↓
Attention
↓
Encoder-Decoder
↓
GAN

## Overview

The experiments in this repository are arranged to help a beginner understand both the mathematics and the practical behavior of each model. Each project separates source implementation, interactive experimentation, and documentation so the learning flow stays clear and professional.

## Learning Path

1. Understand how a single artificial neuron performs threshold-based decisions.
2. Learn how logistic regression models binary outcomes with probabilities.
3. Extend the model to hidden-layer neural networks and backpropagation.
4. Explore visual feature learning through convolutional neural networks.
5. Study transfer learning with pretrained image models.
6. Learn representation learning with autoencoders.
7. Move into sequence models for language problems using RNNs and LSTMs.
8. Understand attention, encoder-decoder systems, and GANs.

## Table of Experiments

| Project | Core Concept | Domain |
| --- | --- | --- |
| MCP Neuron | Artificial Neuron | Logic |
| Logistic Regression | Binary Classification | Tabular |
| Neural Network | Backpropagation | Tabular |
| CNN Classification | Visual Feature Learning | Computer Vision |
| Face Recognition | CNN Classification | Computer Vision |
| Transfer Learning | Pretrained Models | Computer Vision |
| Image Denoising Autoencoder | Representation Learning | Computer Vision |
| Dialogue Generation | LSTM + Attention | NLP |
| Opinion Mining | RNN | NLP |
| Machine Translation | Encoder-Decoder | NLP |
| GAN Image Generation | Generative Modeling | Computer Vision |

## Technologies

- Python 3
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow / Keras
- Jupyter Notebook

## Repository Structure

```text
deep-learning-lab/
├── README.md
├── requirements.txt
├── .gitignore
├── 01-mcculloch-pitts-neuron/
├── 02-logistic-regression/
├── 03-neural-network/
├── 04-cnn-classification/
├── 05-face-recognition/
├── 06-transfer-learning/
├── 07-image-denoising/
├── 08-dialogue-generation/
├── 09-opinion-mining/
├── 10-machine-translation/
├── 11-gan-image-generation/
└── ...
```

## Reproducibility

The repository is organized to separate implementation from experimentation:

- Source code lives in `.py` files.
- Interactive work lives in `.ipynb` notebooks.
- Project explanations live in `README.md` files.

This keeps the repository easy to understand and easy to extend.

## Future Direction

The repository is intentionally structured as a foundation for deeper study. Suggested future topics include:

- PyTorch
- Transformers
- Hugging Face
- Large Language Models
- RAG
- Fine-tuning
- Model deployment
- MLOps

These are future learning directions and are not claims of current implementation in this repository.
