# 📖 SimpleRNN Sentiment Analysis with Hidden State Visualization

## Overview

This project demonstrates how a **Simple Recurrent Neural Network (SimpleRNN)** can be used for **binary sentiment analysis** using TensorFlow and Keras. In addition to training the model, the project explores the internal workings of the network by visualizing:

* Learned word embeddings
* Hidden states generated at each timestep
* Final hidden state used for sentiment prediction

The goal is not only to build a sentiment classifier but also to understand **how an RNN processes sequential text**.

---

# Features

* Binary sentiment classification (Positive / Negative)
* Text preprocessing using Keras `Tokenizer`
* Sequence padding
* Word embedding using the `Embedding` layer
* Sentiment prediction using `SimpleRNN`
* Intermediate Keras model for inspecting layer outputs
* Hidden state visualization for every timestep
* Weight sharing between trained and inspection models

---

# Dataset

The project uses a small manually created dataset consisting of **30 English sentences**.

* **15 Positive sentences**
* **15 Negative sentences**

Example:

### Positive

* I love this product
* Great job, well done
* Amazing quality and value
* Totally recommend to everyone

### Negative

* I hate this product
* Terrible experience overall
* Not worth the money
* Such a waste of time

Labels:

* `1` → Positive
* `0` → Negative

---

# Project Workflow

```text
Text Sentences
       │
       ▼
Tokenizer
       │
       ▼
Integer Sequences
       │
       ▼
Padding
       │
       ▼
Embedding Layer
       │
       ▼
SimpleRNN
       │
       ▼
Dense Layer (Sigmoid)
       │
       ▼
Sentiment Prediction
```

---

# Model Architecture

| Layer     | Output Shape  | Description                          |
| --------- | ------------- | ------------------------------------ |
| Input     | (None, 5)     | Input sequence                       |
| Embedding | (None, 5, 16) | Converts word IDs into dense vectors |
| SimpleRNN | (None, 8)     | Learns sequential information        |
| Dense     | (None, 1)     | Binary sentiment prediction          |

---

# Hyperparameters

| Parameter           |               Value |
| ------------------- | ------------------: |
| Vocabulary Size     |                2000 |
| Embedding Dimension |                  16 |
| RNN Units           |                   8 |
| Batch Size          |                   8 |
| Epochs              |                  25 |
| Loss Function       | Binary Crossentropy |
| Optimizer           |                Adam |

---

# Training

The model is trained using TensorFlow's `model.fit()`.

```python
model.fit(
    X,
    y,
    epochs=25,
    batch_size=8
)
```

After training, the model reaches nearly **100% accuracy** on this small toy dataset.

> **Note:** Since the dataset is intentionally small, the model is expected to overfit. The objective of this project is educational rather than achieving production-level performance.

---

# Intermediate Model

One of the main objectives of this project is to inspect the internal representations learned by the network.

An intermediate Keras model is created to expose the outputs of:

* Embedding layer
* SimpleRNN layer

```python
intermediate_model = Model(
    inputs=model.inputs,
    outputs=[
        model.get_layer("embed").output,
        model.get_layer("simple_rnn").output
    ]
)
```

---

# Inspecting Hidden States

Since the original SimpleRNN uses:

```python
return_sequences=False
```

only the **final hidden state** is returned.

To visualize every hidden state, a second SimpleRNN is created with:

```python
return_sequences=True
```

The trained weights are copied into this inspection model.

```python
trained_weights = model.get_layer("simple_rnn").get_weights()
rnn_seq.set_weights(trained_weights)
```

The inspection model outputs:

```text
(batch_size,
 sequence_length,
 hidden_units)
```

For this project:

```text
(1, 5, 8)
```

meaning:

* 1 sentence
* 5 timesteps
* 8 hidden neurons

---

# Why Copy the Weights?

The inspection model should behave exactly like the trained model.

Copying the weights ensures that both models produce identical hidden representations.

The only difference is:

Original Model

```text
Final Hidden State
```

Inspection Model

```text
Hidden State at Every Timestep
```

---

# Example Output

Sentence:

```text
I love this product
```

Hidden states:

```text
[
[-0.016 -0.011 -0.013 ...]
[ 0.045 -0.046 -0.210 ...]
[-0.010 -0.243 -0.152 ...]
[-0.336 -0.099 -0.260 ...]
[-0.336 -0.099 -0.260 ...]
]
```

The last two rows are identical because the final token is padding (`0`), and the embedding layer uses `mask_zero=True`, causing the RNN to ignore padded positions.

---

# Concepts Covered

* Natural Language Processing (NLP)
* Tokenization
* Vocabulary Building
* Sequence Padding
* Word Embeddings
* Recurrent Neural Networks (RNN)
* Hidden States
* Sequence Modeling
* Intermediate Keras Models
* Weight Sharing
* Model Interpretability

---

# Tech Stack

* Python
* TensorFlow
* Keras
* NumPy

---

# Future Improvements

* Replace `SimpleRNN` with LSTM
* Replace `SimpleRNN` with GRU
* Train on a larger sentiment dataset (IMDb, SST-2)
* Visualize embeddings using PCA or t-SNE
* Plot hidden states as heatmaps
* Compare SimpleRNN, LSTM, and GRU performance
* Add attention mechanisms for interpretability

---

# Learning Outcomes

By completing this project, I gained a deeper understanding of:

* How text is converted into numerical sequences.
* How embedding layers learn semantic word representations.
* How SimpleRNN updates hidden states while reading a sentence.
* The difference between `return_sequences=True` and `return_sequences=False`.
* How to build intermediate Keras models for inspecting internal layer outputs.
* How to visualize hidden representations for better model interpretability.

---

# License

This project is intended for educational and learning purposes. Feel free to use, modify, and build upon it with appropriate attribution.
