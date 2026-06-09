# Simple Neural Network - Purchase Prediction

## 📋 Project Overview

This project implements a **simple neural network from scratch** using only NumPy to predict whether a user will purchase a product based on their demographic information. The neural network is built without using high-level deep learning frameworks, demonstrating the fundamental concepts of forward propagation, backpropagation, and gradient descent.

## 🎯 Objective

Predict whether a user will make a purchase (binary classification: 0 = No Purchase, 1 = Purchase) based on:
- **Age**: User's age in years
- **Estimated Salary**: User's estimated annual salary

## 📊 Dataset

**File**: `Social_Network_Ads.csv`

### Dataset Description:
- **Total Records**: 400 samples
- **Features**:
  - `Age`: Numeric (continuous)
  - `EstimatedSalary`: Numeric (continuous)
- **Target Variable**:
  - `Purchased`: Binary (0 or 1)
    - `0`: User did **not** purchase
    - `1`: User **purchased**

### Sample Data:
```
Age  | EstimatedSalary | Purchased
-----|-----------------|----------
19   | 19000           | 0
35   | 20000           | 0
32   | 150000          | 1
47   | 25000           | 1
```

## 🏗️ Neural Network Architecture

### Network Structure:
```
Input Layer (2 features) 
    ↓
Hidden Layer (10 neurons with Sigmoid activation)
    ↓
Output Layer (1 neuron with Sigmoid activation)
```

### Architecture Details:
- **Input Layer**: 2 neurons (Age, EstimatedSalary)
- **Hidden Layer**: 10 neurons
  - Activation: Sigmoid function
  - Weights initialized: Random normal distribution × 0.1
  - Biases initialized: Zeros
- **Output Layer**: 1 neuron
  - Activation: Sigmoid function (for binary classification)
  - Outputs probability between 0 and 1

## 🔧 Implementation Details

### Dependencies:
```python
- numpy: For numerical computations
- pandas: For data loading and manipulation
- sklearn.model_selection.train_test_split: For splitting data into train/test sets
- sklearn.preprocessing.StandardScaler: For feature normalization
```

### Key Components:

#### 1. **Data Preprocessing**
- **Train-Test Split**: Data is split into 80% training (320 samples) and 20% testing (80 samples) with random_state=42
- **Feature Scaling**: StandardScaler is applied to normalize features (Age and EstimatedSalary)
- This ensures both features contribute equally to the model, preventing features with larger values from dominating

#### 2. **Activation Function**
```python
Sigmoid: σ(x) = 1 / (1 + e^(-x))
Derivative: σ'(x) = σ(x) × (1 - σ(x))
```

#### 3. **Loss Function**
- **Binary Cross-Entropy Loss**:
```
J = -1/N × Σ[y × log(ŷ) + (1-y) × log(1-ŷ)]
```
- Small epsilon (1e-15) added to prevent log(0)

#### 4. **Training Process**
- **Algorithm**: Gradient Descent with Backpropagation
- **Learning Rate**: 1.0
- **Epochs**: 4000
- **Batch Size**: Full batch (320 training samples)

### Training Steps:
1. **Forward Pass**:
   - Calculate hidden layer: z1 = X·W1 + b1, a1 = sigmoid(z1)
   - Calculate output: z2 = a1·W2 + b2, a2 = sigmoid(z2)
   
2. **Loss Calculation**:
   - Compute binary cross-entropy loss
   
3. **Backward Pass** (Backpropagation):
   - Calculate gradients for output layer (dW2, db2)
   - Backpropagate error to hidden layer (dW1, db1)
   
4. **Weight Update**:
   - Update weights and biases using gradient descent

## 🚀 How to Run

### Prerequisites:
```bash
pip install numpy pandas scikit-learn
```

### Execution:
```bash
python main.py
```

### Expected Output:
The script will display:
- Loss value every 100 epochs during training
- Final loss and accuracy on the training set
- Model accuracy on the testing set

Example output:
```
Epoch    0, Loss: 0.693147
Epoch  100, Loss: 0.427158
Epoch  200, Loss: 0.362841
...
Epoch 3900, Loss: 0.241563

--- TRAINING SET RESULTS ---
Final Loss on Training Set: 0.220348
Model Accuracy on Training Set: 92.50%

--- TESTING SET RESULTS ---
Model Accuracy on Testing Set: 90.00%
```

## 📈 Model Evaluation

### Prediction Logic:
- If output probability ≥ 0.5 → Predict **Purchase (1)**
- If output probability < 0.5 → Predict **No Purchase (0)**

### Accuracy Calculation:
```python
Accuracy = (Correct Predictions / Total Predictions) × 100%
```

## 🧠 Learning Concepts Demonstrated

This project showcases fundamental deep learning concepts:

1. **Neural Network Layers**: Multi-layer perceptron architecture
2. **Forward Propagation**: Computing predictions from inputs
3. **Backpropagation**: Calculating gradients using chain rule
4. **Gradient Descent**: Optimizing weights to minimize loss
5. **Binary Classification**: Sigmoid activation for probability output
6. **Feature Scaling**: Normalizing inputs for better convergence
7. **Weight Initialization**: Preventing vanishing/exploding gradients

## 📚 Key Takeaways

- ✅ Built entirely from scratch using NumPy (no TensorFlow/PyTorch)
- ✅ Demonstrates core concepts of neural networks
- ✅ Shows how backpropagation works mathematically
- ✅ Achieves good accuracy (~90%) on purchase prediction
- ✅ Practical example of binary classification

## 🔍 Potential Improvements

1. **Validation Set**: Add a separate validation set for hyperparameter tuning
2. **Regularization**: Add L2 regularization to prevent overfitting
3. **Learning Rate Scheduling**: Decrease learning rate over time
4. **Different Activations**: Try ReLU for hidden layers
5. **Cross-Validation**: Implement k-fold cross-validation
6. **Hyperparameter Tuning**: Optimize neurons, learning rate, epochs
7. **Visualization**: Add loss curves and decision boundary plots
8. **Mini-batch Training**: Implement mini-batch gradient descent for better generalization

## 📝 Notes

- This is an educational implementation focusing on understanding neural network fundamentals
- For production use, consider using established frameworks like TensorFlow or PyTorch
- The model uses the entire dataset for training without a validation or test set

---

*This project demonstrates a fundamental understanding of neural networks and serves as a foundation for more advanced deep learning projects.*
