# Breast Cancer Diagnosis Classification Analysis

## 📊 Project Overview

This project presents a comprehensive classification analysis of the **Breast Cancer Wisconsin (Diagnostic)** dataset. The goal is to predict whether a breast mass is **benign** or **malignant** based on various cell characteristics. Multiple classification algorithms are implemented and compared to determine the best-performing model for accurate cancer diagnosis.

## 📁 Dataset Information

The dataset contains 683 observations from breast cancer biopsies collected at the University of Wisconsin Hospitals, Madison, from Dr. William H. Wolberg.

### Features:
- **Clump Thickness** - Thickness of cell clumps (1-10)
- **Uniformity of Cell Size** - Consistency in cell size (1-10)
- **Uniformity of Cell Shape** - Consistency in cell shape (1-10)
- **Marginal Adhesion** - How cells stick together (1-10)
- **Single Epithelial Cell Size** - Size of epithelial cells (1-10)
- **Bare Nuclei** - Nuclei not surrounded by cytoplasm (1-10)
- **Bland Chromatin** - Texture of cell nucleus (1-10)
- **Normal Nucleoli** - Nucleoli characteristics (1-10)
- **Mitoses** - Cell division rate (1-10)

### Target Variable:
- **Class** - Diagnosis (2 = Benign, 4 = Malignant)

### Dataset Characteristics:
- **Total Samples**: 683 (after removing missing values)
- **Benign Cases**: ~65%
- **Malignant Cases**: ~35%
- **Source**: University of Wisconsin Hospitals, Madison

## 🗂️ Project Structure

```
.
├── Data.csv                          # Breast Cancer Wisconsin dataset
├── 01_data_visualisation.ipynb      # Exploratory Data Analysis & Visualization
├── 02_stats_models.ipynb            # Statistical Analysis using Statsmodels
├── 03_logistic_regression.ipynb     # Logistic Regression Classification
├── 04_naive_bayes.ipynb             # Naive Bayes Classifier
├── 05_k_nearest_neighbours.ipynb    # K-Nearest Neighbors (KNN) Classifier
└── README.md                         # Project documentation
```

## 📓 Notebooks Description

### 1️⃣ Data Visualization (`01_data_visualisation.ipynb`)
- **Purpose**: Exploratory Data Analysis (EDA)
- **Key Activities**:
  - Visualizing distributions of features for benign vs malignant cases
  - Analyzing correlation patterns between features
  - Creating comprehensive visualizations to understand feature importance
  - Box plots and histograms for each feature by class
  - Identifying patterns that distinguish benign from malignant tumors

### 2️⃣ Statistical Models (`02_stats_models.ipynb`)
- **Purpose**: Statistical analysis using Statsmodels library
- **Key Activities**:
  - Logistic regression using Statsmodels for statistical inference
  - Detailed statistical summary including:
    - Coefficient estimates
    - p-values for feature significance
    - Odds ratios
    - Confidence intervals
    - Log-likelihood and pseudo R-squared
  - Statistical hypothesis testing for feature importance

### 3️⃣ Logistic Regression (`03_logistic_regression.ipynb`)
- **Purpose**: Scikit-learn implementation of Logistic Regression
- **Key Activities**:
  - Data preprocessing and train-test split (80-20)
  - Model training using LogisticRegression
  - Predictions and probability estimates
  - Confusion matrix visualization
  - Performance metrics calculation:
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - ROC-AUC Score
  - ROC curve visualization

### 4️⃣ Naive Bayes (`04_naive_bayes.ipynb`)
- **Purpose**: Probabilistic classification using Naive Bayes
- **Key Activities**:
  - Train-test split (80-20)
  - Gaussian Naive Bayes implementation
  - Predictions based on Bayes theorem
  - Confusion matrix analysis
  - Performance metrics evaluation
  - Comparison with Logistic Regression

### 5️⃣ K-Nearest Neighbors (`05_k_nearest_neighbours.ipynb`)
- **Purpose**: Instance-based classification using KNN
- **Key Activities**:
  - Train-test split (80-20)
  - Feature scaling/normalization
  - Model training with different K values
  - K-value optimization (elbow method)
  - Confusion matrix visualization
  - Performance metrics calculation
  - Distance metric analysis

### 6️⃣ Additional Classification Algorithms (Upcoming)
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Machines (XGBoost, LightGBM)
- Neural Networks
- Ensemble Methods

## 📈 Performance Metrics

All models will be evaluated and compared using the following metrics:

| Metric | Description | Goal |
|--------|-------------|------|
| **Accuracy** | Overall correct predictions | Higher is better |
| **Precision** | True positives / (True positives + False positives) | Higher is better |
| **Recall** | True positives / (True positives + False negatives) | Higher is better |
| **F1-Score** | Harmonic mean of precision and recall | Higher is better |
| **ROC-AUC** | Area under ROC curve | Higher is better (max 1.0) |
| **Specificity** | True negatives / (True negatives + False positives) | Higher is better |

**Note**: For medical diagnosis, **Recall (Sensitivity)** is particularly important as we want to minimize false negatives (missing malignant cases).

## 📊 Model Comparison (To Be Updated)

### Classification Report by Model

|     Model     | Class | Precision | Recall | F1-Score | Support | Accuracy | ROC-AUC |
|---------------|-------|-----------|--------|----------|---------|----------|---------|
| **Logistic Regression** | Benign (2) | 0.95 | 0.97 | 0.96 | 107 | 0.95 | 0.95 |
|  | Malignant (4) | 0.95 | 0.92 | 0.94 | 64 |  |  |
| **Naive Bayes** | Benign (2) | 0.99 | 0.93 | 0.96 | 107 | 0.95 | 0.95 |
|  | Malignant (4) | 0.90 | 0.98 | 0.94 | 64 |  |  |
| **K-Nearest Neighbors** | Benign (2) | 0.97 | 0.97 | 0.97 | 107 | 0.96 | 0.96 |
|  | Malignant (4) | 0.95 | 0.95 | 0.95 | 64 |  |  |
| **Support Vector Machine** | Benign (2) | TBD | TBD | TBD | TBD | TBD | TBD |
|  | Malignant (4) | TBD | TBD | TBD | TBD |  |  |
| **Decision Tree** | Benign (2) | 0.94 | 0.95 | 0.94 | 107 | 0.93 | 0.92 |
|  | Malignant (4) | 0.92 | 0.89 | 0.90 | 64 |  |  |
| **Random Forest** | Benign (2) | 0.97| 0.96 | 0.97 | 107 | 0.96 | 0.96 |
|  | Malignant (4) | 0.94 | 0.95 | 0.95 | 64 |  |  |

**Note**: 
- **Benign (2)**: Non-cancerous tumors
- **Malignant (4)**: Cancerous tumors
- **Support**: Number of samples in each class in the test set
- For malignant cases, **Recall** is the most critical metric (minimizing false negatives)


## 🔍 Key Insights (So Far)

Based on initial analysis:
- **Uniformity of Cell Size** and **Uniformity of Cell Shape** are strong predictors of malignancy
- **Bare Nuclei** characteristic shows significant difference between benign and malignant cases
- **Clump Thickness** is notably higher in malignant tumors
- Most features show clear separation between the two classes
- Feature correlation analysis reveals multicollinearity among some variables

## 🛠️ Technologies & Libraries

- **Python 3.12**
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: statsmodels
- **Machine Learning**: scikit-learn
- **Performance Metrics**: sklearn.metrics
- **Environment**: Jupyter Notebook

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn statsmodels scikit-learn jupyter
```

### Running the Notebooks
1. Clone or download this repository
2. Ensure `Data.csv` is in the same directory
3. Open Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
4. Run the notebooks in sequential order (01 → 02 → 03 → 04 → 05 → ...)

## 📝 Future Work

- [ ] Implement Support Vector Machine (SVM) with different kernels
- [ ] Add Decision Tree and Random Forest classifiers
- [ ] Explore ensemble methods (Voting, Stacking, Boosting)
- [ ] Perform feature engineering and selection
- [ ] Implement cross-validation for robust model evaluation
- [ ] Add hyperparameter tuning using GridSearchCV
- [ ] Handle class imbalance if needed (SMOTE, class weights)
- [ ] Create a final ensemble model combining best classifiers

## 👤 Author

A Data Science practice project focused on mastering classification techniques and contributing to healthcare analytics.

## 📚 Credits

- **Dataset**: [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(diagnostic)) from UCI Machine Learning Repository
- **Original Donor**: Dr. William H. Wolberg, University of Wisconsin Hospitals, Madison
- **Course**: Inspired by Udemy course [Machine Learning A-Z](https://www.udemy.com/course-dashboard-redirect/?course_id=950390/)

## 📄 License

This is an educational project for learning purposes.

---

**Last Updated**: May 2026
