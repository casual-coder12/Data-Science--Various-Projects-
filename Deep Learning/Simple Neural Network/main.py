import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Social_Network_Ads.csv')
df = pd.read_csv(csv_path)

# Extracting the relevant features and target variable
X = df.iloc[:, :-1].values  # All columns except the last one as features
y = df.iloc[:, -1].values.reshape(-1, 1)  # The last column as the target variable

# Spliting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implementing feature scaling using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

N = X_train.shape[0]

# --- INITIALIZATION OF NEURAL NETWORK ---
# Layer 1 (Hidden layer - 10 neurons)
W1 = np.random.randn(X_train.shape[1], 10) * 0.1
b1 = np.zeros((1, 10))

# Layer 2 (Output layer - 1 neuron)
W2 = np.random.randn(10, 1) * 0.1
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    # Derivative of the sigmoid function expressed in terms of the activation 'a'
    return a * (1 - a)

lr = 0.5  # Learning rate

for epoch in range(10000):
    # --- FORWARD PASS ---
    # Layer 1
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)  # Activation of the hidden layer
    
    # Layer 2
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # Final output of the network

    # Calculating the cost function (Loss) using the output a2
    eps = 1e-15
    J = np.mean(-(y_train * np.log(a2 + eps) + (1 - y_train) * np.log(1 - a2 + eps)))

    # --- BACKWARD PASS (Backpropagation) ---
    # Gradients for Layer 2 (Output layer)
    dz2 = a2 - y_train
    dW2 = np.dot(a1.T, dz2) / N
    db2 = np.sum(dz2, axis=0, keepdims=True) / N

    # Gradients for Layer 1 (Hidden layer)
    # Error is backpropagated through W2 and multiplied by the derivative of the hidden layer activation
    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X_train.T, dz1) / N
    db1 = np.sum(dz1, axis=0, keepdims=True) / N

    # --- UPDATE WEIGHTS AND BIASES ---
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}, Loss: {J:.6f}')

# --- EVALUATION OF THE MODEL ---

# Training set evaluation

# 1. Forward pass with the trained weights on training set to get the predicted probabilities
final_z1 = np.dot(X_train, W1) + b1
final_a1 = sigmoid(final_z1)
final_z2 = np.dot(final_a1, W2) + b2
final_a2 = sigmoid(final_z2)

# 2. Converting probabilities to classes (0 or 1)
y_pred = (final_a2 >= 0.5).astype(int)

# 3. Calculating accuracy percentage for the training set
accuracy = np.mean(y_pred == y_train) * 100

print("\n--- TRAINING SET RESULTS ---")
print(f"Final Loss on Training Set: {J:.6f}")
print(f"Model Accuracy on Training Set: {accuracy:.2f}%")

# Testing set evaluation

# 1. Forward pass with the trained weights on testing set to get the predicted probabilities
test_set_z1 = np.dot(X_test, W1) + b1
test_set_a1 = sigmoid(test_set_z1)
test_set_z2 = np.dot(test_set_a1, W2) + b2
test_set_a2 = sigmoid(test_set_z2)

# 2. Converting probabilities to classes (0 or 1)
y_pred_test = (test_set_a2 >= 0.5).astype(int)

# 3. Calculating accuracy percentage for the testing set
accuracy_test = np.mean(y_pred_test == y_test) * 100

print("\n--- TESTING SET RESULTS ---")
print(f"Model Accuracy on Testing Set: {accuracy_test:.2f}%")
