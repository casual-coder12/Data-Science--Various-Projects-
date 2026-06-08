import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'Social_Network_Ads.csv')
df = pd.read_csv(csv_path)

# Implementing feature scaling using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df.iloc[:, :-1])

y = df.iloc[:, -1].values.reshape(-1, 1)

N = X.shape[0]

# --- INITIALIZATION OF NEURAL NETWORK ---
# Layer 1 (Hidden layer - 10 neurons)
W1 = np.random.randn(X.shape[1], 10) * 0.1
b1 = np.zeros((1, 10))

# Layer 2 (Output layer - 1 neuron)
W2 = np.random.randn(10, 1) * 0.1
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(a):
    # Derivative of the sigmoid function expressed in terms of the activation 'a'
    return a * (1 - a)

lr = 1  # Learning rate

for epoch in range(4000):
    # --- FORWARD PASS ---
    # Layer 1
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)  # Activation of the hidden layer
    
    # Layer 2
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)  # Final output of the network

    # Calculating the cost function (Loss) using the output a2
    eps = 1e-15
    J = np.mean(-(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps)))

    # --- BACKWARD PASS (Backpropagation) ---
    # Gradients for Layer 2 (Output layer)
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / N
    db2 = np.sum(dz2, axis=0, keepdims=True) / N

    # Gradients for Layer 1 (Hidden layer)
    # Error is backpropagated through W2 and multiplied by the derivative of the hidden layer activation
    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1) / N
    db1 = np.sum(dz1, axis=0, keepdims=True) / N

    # --- UPDATE WEIGHTS AND BIASES ---
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 100 == 0:
        print(f'Epoch {epoch:4d}, Loss: {J:.6f}')

# --- EVALUATION OF THE MODEL ---

# 1. Final forward pass with the trained weights
final_z1 = np.dot(X, W1) + b1
final_a1 = sigmoid(final_z1)
final_z2 = np.dot(final_a1, W2) + b2
final_a2 = sigmoid(final_z2)

# 2. Converting probabilities to classes (0 or 1)
y_pred = (final_a2 >= 0.5).astype(int)

# 3. Calculating accuracy percentage
accuracy = np.mean(y_pred == y) * 100

print("\n--- RESULTS ---")
print(f"Final Loss: {J:.6f}")
print(f"Model Accuracy: {accuracy:.2f}%")
