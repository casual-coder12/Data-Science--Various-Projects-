# Neural Network with Keras & TensorFlow (Custom Training Loop with `GradientTape`)

This project implements a simple feed-forward neural network for a binary classification task using **TensorFlow** and **Keras**, with a **custom training loop** built using `tf.GradientTape()` instead of the standard `model.fit()` API. It's designed as a learning exercise to understand what happens "under the hood" during model training — including gradient computation, optimizer updates, and metric tracking.

## Project Overview

The notebook (`nn_keras.ipynb`) walks through the full deep learning workflow:

1. **Data Generation** — Synthetic non-linear data using `make_moons` from scikit-learn.
2. **Data Splitting** — Train / validation / test split (≈70% / 15% / 15%).
3. **Data Pipeline** — Building efficient input pipelines with `tf.data.Dataset`.
4. **Model Definition** — A `Sequential` Keras model with Dense layers.
5. **Custom Training Loop** — Manual forward pass, loss computation, and backpropagation using `tf.GradientTape()`.
6. **Evaluation** — Accuracy calculation on the held-out test set.
7. **Visualization** — Decision boundary plotting to visually assess model performance.

## Dataset

The dataset is generated with `sklearn.datasets.make_moons`, producing two interleaving half-moon-shaped clusters — a classic non-linearly separable binary classification problem.

```python
X_raw, y_raw = make_moons(n_samples=1200, noise=0.2, random_state=42)
```

- **Samples:** 1200
- **Features:** 2 (x1, x2 coordinates)
- **Target:** Binary label (0 or 1)
- **Noise:** 0.2 (adds Gaussian noise for a more realistic, less separable dataset)

## Data Splitting Strategy

The data is split in two stages to achieve a **70% train / 15% validation / 15% test** split:

| Step | Split | Result |
|------|-------|--------|
| A | 85% (train+val) / 15% (test) | Isolates the final test set |
| B | 82.4% (train) / 17.6% (val) of the remaining 85% | Produces the 70/15/15 ratio |

Stratified sampling (`stratify=y_raw`) is used to preserve class balance across all splits.

## Input Pipeline

Data is loaded into TensorFlow using `tf.data.Dataset` for efficient batching:

```python
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1024).batch(32)
val_dataset   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
test_dataset  = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)
```

Only the training set is shuffled, ensuring validation/test evaluation remains consistent across epochs.

## Model Architecture

A simple fully-connected (Dense) feed-forward network built with `tf.keras.Sequential`:

| Layer | Units | Activation |
|-------|-------|------------|
| Dense (Input) | 32 | ReLU |
| Dense (Hidden) | 16 | ReLU |
| Dense (Output) | 1 | Sigmoid |

- **Input shape:** (2,) — matches the 2D feature space of the dataset.
- **Output:** A single probability value for binary classification.

## Optimizer & Loss

- **Learning Rate Schedule:** `ExponentialDecay` — starts at `0.01` and decays by a factor of `0.9` every `1000` steps.
- **Optimizer:** `Adam`
- **Loss Function:** `BinaryCrossentropy`
- **Metric:** `BinaryAccuracy` (tracked separately for training and validation)

```python
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01, decay_steps=1000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
loss_fn = tf.keras.losses.BinaryCrossentropy()
```

## Custom Training Loop (`GradientTape`)

Instead of `model.fit()`, this project manually implements the training and validation steps to demonstrate how gradients are computed and applied:

```python
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_accuracy.update_state(y_batch, predictions)
    return loss
```

A similar `val_step` function evaluates the model without updating weights (`training=False`, no gradient application).

The main training loop runs for **50 epochs**, printing training/validation loss and accuracy after each epoch.

## Evaluation

After training, the model is evaluated on the unseen **test set**:

```python
predictions = model.predict(X_test)
predictions_cls = (predictions > 0.5).astype(np.float32)

test_accuracy_b = tf.keras.metrics.BinaryAccuracy()
test_accuracy_b.update_state(y_test, predictions_cls)
```

This produces the final test accuracy, reported as a single metric.

## Visualization

A helper function `plot_decision_boundary()` plots the learned decision boundary over a mesh grid, overlaid with the actual data points — providing an intuitive visual check of how well the model separates the two classes on both the training and test sets.

## Requirements

- Python 3.12
- `tensorflow`
- `numpy`
- `matplotlib`
- `scikit-learn`

Install dependencies (from the repository root):

```bash
pip install -r requirements.txt
```

## Running the Notebook

1. Open `nn_keras.ipynb` in Jupyter Notebook / JupyterLab / VS Code.
2. Run all cells sequentially from top to bottom.
3. Review the printed epoch-by-epoch metrics and the final test accuracy.
4. Inspect the decision boundary plots at the end of the notebook.

## Key Takeaways

- Demonstrates how to build a **custom training loop** in TensorFlow using `GradientTape`, offering full control over the forward pass, loss computation, and gradient updates.
- Shows proper **train/validation/test splitting** practices with stratification.
- Illustrates the use of `tf.data.Dataset` for efficient, scalable data pipelines.
- Highlights the difference between using high-level Keras APIs (`model.fit()`) and low-level manual training for educational purposes.
