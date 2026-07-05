# Neural Network with Keras & TensorFlow (Custom `GradientTape` Training Loop)

A binary classification neural network built with **TensorFlow/Keras**, trained using a **custom training loop** (`tf.GradientTape()`) instead of `model.fit()`. This is a learning exercise to understand what happens under the hood during training.

## Overview

`nn_keras.ipynb` covers:

1. **Data** — Synthetic non-linear data via `sklearn.datasets.make_moons` (1200 samples, 2 features, binary label).
2. **Split** — Stratified 70% train / 15% validation / 15% test.
3. **Pipeline** — Batched with `tf.data.Dataset` (train set shuffled).
4. **Model** — `Sequential`: Dense(32, relu) → Dense(16, relu) → Dense(1, sigmoid).
5. **Optimization** — Adam optimizer with `ExponentialDecay` learning rate schedule; `BinaryCrossentropy` loss.
6. **Training** — Custom `train_step`/`val_step` functions using `tf.GradientTape()` to compute and apply gradients manually, run for 50 epochs.
7. **Evaluation** — Final accuracy computed on the held-out test set.
8. **Visualization** — Decision boundary plotted over train and test data.

## Requirements

```bash
pip install -r requirements.txt
```
(`tensorflow`, `numpy`, `matplotlib`, `scikit-learn`)

## Running

Open `nn_keras.ipynb` and run all cells top to bottom to see training progress (loss/accuracy per epoch), final test accuracy, and decision boundary plots.

## Key Takeaway

Demonstrates how to manually implement forward pass, loss computation, and gradient updates in TensorFlow — instead of relying on the high-level `model.fit()` API — for a deeper understanding of the training process.
