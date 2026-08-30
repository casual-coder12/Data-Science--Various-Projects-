# LeNet-5 Benchmarking: MNIST vs. CIFAR-10

A historical evaluation of the **LeNet-5** CNN architecture (LeCun et al., 1998) trained using Stochastic Gradient Descent (SGD) on two foundational computer vision benchmarks.

## 🚀 Quick Results

| Dataset | Type | Input Dim | Test Accuracy |
| :--- | :--- | :---: | :---: |
| **MNIST** | Handwritten Digits | $32 \times 32 \times 1$ | **96.65%** |
| **CIFAR-10** | Real-world Objects | $32 \times 32 \times 3$ | **45.48%** |

## 💡 Key Findings
* **MNIST (96.65%):** LeNet-5 easily extracts spatial features from simple, single-channel geometric patterns.
* **CIFAR-10 (45.48%):** The shallow depth and low feature map capacity ($6/16$ channels) limit the model's ability to generalize across high intra-class color variations and complex backgrounds.

## 🛠️ How to Run
1. Clone the repository.
2. Install requirements: `pip install tensorflow matplotlib scikit-learn datasets`.
3. Run the main notebook: `LeNet-5.ipynb`.