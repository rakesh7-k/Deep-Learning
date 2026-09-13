# Machine Translation with Encoder-Decoder LSTM

This experiment builds a small educational machine translation system using an English-to-French encoder-decoder model based on LSTMs.

## Overview

Machine translation maps an input sentence in one language to a target sentence in another language. In this experiment, the scope is intentionally small and teaching-oriented.

## Core Idea

The encoder reads the source sentence and forms a context vector. The decoder then generates the translated sentence one token at a time, using teacher forcing during training.

## Architecture

```text
English -> Encoder LSTM -> Context Representation -> Decoder LSTM -> French
```

## Mathematical Foundation

- LSTM for context modeling in sequences
- Decoder generation of output tokens
- Teacher forcing to stabilize training

## Dataset

A tiny parallel English-French dataset is used so the example remains easy to study and run in a student environment.

## Implementation

The notebook tokenizes the sentences, pads the inputs, builds the encoder-decoder network, trains it, and demonstrates translation inference on sample phrases.

## Experiments

- Tokenize and align English-French pairs.
- Train the translation model.
- Generate examples at inference time.
- Explain the limits of tiny datasets.

## Results

The system demonstrates learning behavior on a toy dataset, but the translations are not production-ready.

## Technical Insights

A tiny dataset cannot capture the full complexity of real language translation. This project is valuable for learning the architecture, while its performance is intentionally limited by the dataset size and training setup.

## Limitations

- Small corpus restricts generalization.
- Not suitable for production translation quality.
- Sequence generation is sensitive to training setup and data size.

## Possible Improvements

- Increase dataset size.
- Use attention-rich models.
- Compare with transformers and pretrained multilingual models.

## Reproducibility

Run the notebook in the project folder and keep the dataset intentionally small so the experiment remains educational and manageable.

## Technologies

- Python 3
- TensorFlow / Keras
- NumPy
- Matplotlib
- Jupyter Notebook
