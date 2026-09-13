# Opinion Mining Using RNN

This experiment uses a recurrent neural network for sentiment analysis on the IMDB movie review dataset. The model learns patterns in text to classify reviews as positive or negative.

## Overview

Opinion mining, or sentiment analysis, predicts the emotional tone of a text. In this example, the task is binary classification: positive versus negative sentiment.

## Core Idea

A recurrent neural network processes a sequence of words and updates its internal state over time, allowing it to build context from earlier tokens.

## Architecture

```text
Text -> Tokenization -> Embedding -> SimpleRNN -> Sigmoid -> Sentiment
```

## Mathematical Foundation

- Embedding layers map tokens to dense vectors.
- RNN states maintain sequence context.
- Sigmoid output gives a probability for positive sentiment.

## Dataset

The IMDB dataset is used for a standard sentiment-classification example.

## Implementation

The notebook covers dataset loading, preprocessing, tokenization, model building, training, evaluation, and sample prediction analysis.

## Experiments

- Explore a sample of reviews.
- Tokenize and pad sequences.
- Train the RNN.
- Plot training curves.
- Inspect predicted sentiment labels.

## Results

The model learns a useful sentiment signal from review text and demonstrates basic text classification behavior.

## Technical Insights

This experiment introduces the idea that words in a sequence are not independent. Recurrent models maintain memory over time, making them useful for text-based understanding tasks.

## Limitations

- A basic RNN can struggle with long-range dependencies.
- The dataset is educational and modest in scale.
- Real sentiment analysis usually benefits from richer architectures and larger data.

## Possible Improvements

- Replace SimpleRNN with LSTM or GRU.
- Add pretrained embeddings.
- Use more advanced text preprocessing and regularization.

## Reproducibility

Install the dependencies and run the notebook sequentially in this folder. The dataset is a standard benchmark that can be downloaded through Keras.

## Technologies

- Python 3
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
