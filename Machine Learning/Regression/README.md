# Combined Cycle Power Plant Regression Analysis

## 📊 Project Overview

This project presents a comprehensive regression analysis of the **Combined Cycle Power Plant** dataset. The goal is to predict the net hourly electrical energy output (PE) of the power plant using various environmental features. Multiple regression algorithms are implemented and compared to determine the best-performing model.

## 📁 Dataset Information

The dataset contains 9568 observations collected from a Combined Cycle Power Plant over 6 years (2006-2011).

### Features:
- **AT** - Ambient Temperature (°C)
- **V** - Exhaust Vacuum (cm Hg)
- **AP** - Ambient Pressure (millibar)
- **RH** - Relative Humidity (%)

### Target Variable:
- **PE** - Net hourly electrical energy output (MW)

## 🗂️ Project Structure

```
.
├── Data.csv                          # Combined Cycle Power Plant dataset
├── 01_data_visualisation.ipynb      # Exploratory Data Analysis & Visualization
├── 02_stats_models.ipynb            # Statistical Analysis using Statsmodels
├── 03_linear_regression.ipynb       # Multiple Linear Regression
├── 04_polynomial_regression.ipynb   # Polynomial Regression (upcoming)
└── README.md                         # Project documentation
```

## 📓 Notebooks Description

### 1️⃣ Data Visualization (`01_data_visualisation.ipynb`)
- **Purpose**: Exploratory Data Analysis (EDA)
- **Key Activities**:
  - Visualizing relationships between features and target variable using scatter plots
  - Analyzing correlation patterns with lmplot
  - Creating comprehensive pairplots to understand feature interactions
  - Identifying trends and patterns in the data

### 2️⃣ Statistical Models (`02_stats_models.ipynb`)
- **Purpose**: Statistical analysis using Statsmodels library
- **Key Activities**:
  - Ordinary Least Squares (OLS) regression implementation
  - Detailed statistical summary including:
    - Coefficient estimates
    - p-values for feature significance
    - R-squared and Adjusted R-squared
    - F-statistic
    - Confidence intervals
  - Statistical inference and hypothesis testing

### 3️⃣ Multiple Linear Regression (`03_linear_regression.ipynb`)
- **Purpose**: Scikit-learn implementation of Multiple Linear Regression
- **Key Activities**:
  - Train-test split (80-20)
  - Model training using LinearRegression
  - Predictions visualization (actual vs predicted)
  - Residual analysis and distribution
  - Performance metrics calculation:
    - R² Score
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Relative MAE (percentage)

### 4️⃣ Polynomial Regression (`04_polynomial_regression.ipynb`)
- **Purpose**: Capture non-linear relationships
- **Key Activities**:
  - Train-test split (80-20)
  - Feature transformation using polynomial features
  - Model training with different polynomial degrees
  - Predictions visualization (actual vs predicted)
  - Residual analysis and distribution
  - Performance metrics calculation:
  - Overfitting analysis

### 5️⃣ Additional Regression Models (Upcoming)
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- Elastic Net (L1 + L2 regularization)
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression

## 📈 Performance Metrics

All models will be evaluated and compared using the following metrics:

| **R² Score** | Coefficient of determination | Higher is better (max 1.0) |
| **MAE** | Mean Absolute Error | Lower is better |
| **MSE** | Mean Squared Error | Lower is better |
| **RMSE** | Root Mean Squared Error | Lower is better |
| **Relative MAE** | MAE as percentage of mean | Lower is better |

|----------|------|-------|------|---------|---------------|
| R² Score |  MAE |  MSE  | RMSE | Rel MAE |     Model     |
|----------|------|-------|------|---------|---------------|
|  0.9325  | 3.56 | 19.73 | 4.44 |  0.78%  | Multiple Linear Regression |
|  0.9455  | 3.14 | 15.93 | 3.99 |  0.69%  | Polynomial Regression |

## 🔍 Key Insights (So Far)

Based on initial analysis:
- **Ambient Temperature (AT)** shows a strong negative correlation with Power Output (PE)
- **Exhaust Vacuum (V)** demonstrates a positive linear relationship with PE
- **Ambient Pressure (AP)** shows a positive correlation with PE
- **Relative Humidity (RH)** has a weak negative relationship with PE

## 🛠️ Technologies & Libraries

- **Python 3.12**
- **Data Manipulation**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Statistical Analysis**: statsmodels
- **Machine Learning**: scikit-learn
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
4. Run the notebooks in sequential order (01 → 02 → 03 → ...)

## 📝 Future Work

- [ ] Implement polynomial regression with degree optimization
- [ ] Add regularization techniques (Ridge, Lasso, Elastic Net)
- [ ] Explore ensemble methods (Random Forest, Gradient Boosting)
- [ ] Perform feature engineering and selection
- [ ] Implement cross-validation for robust model evaluation
- [ ] Create final visualization dashboard comparing all models
- [ ] Add hyperparameter tuning using GridSearchCV

## 👤 Author

A Data Science practice project focused on mastering regression techniques.

## Credits

Data from Kaggle (https://www.kaggle.com/datasets/rinichristy/combined-cycle-power-plant-data-set-uci-data)
Inspired by Udemy course "Machine Learning A-Z" (https://www.udemy.com/course-dashboard-redirect/?course_id=950390/)

## 📄 License

This is an educational project for learning purposes.

---

**Last Updated**: May 2026
