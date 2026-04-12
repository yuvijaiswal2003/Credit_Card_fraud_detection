# Credit Card Fraud Detection

## Project Overview
Credit card fraud is a major issue for financial institutions and their customers. With the increasing number of transactions, detecting fraudulent activity has become crucial. This project aims to develop a machine learning model capable of identifying fraudulent credit card transactions accurately and minimizing false positives. By leveraging historical transaction data, the model will identify patterns and anomalies that suggest fraudulent behavior.

## Project Objectives
- Build a machine learning model that can accurately detect fraudulent credit card transactions.
- Minimize false positives while ensuring high accuracy in identifying fraudulent activities.
- Evaluate and compare multiple machine learning algorithms to find the best performing model.

## Key Steps

### 1. Data Collection
- Collected credit card transaction data, including both fraudulent and non-fraudulent transactions, from publicly available datasets (e.g., Kaggle or UCI Machine Learning Repository).

### 2. Data Preprocessing
- Cleaned the data by handling missing values and irrelevant information.
- Performed feature engineering to create additional features that can improve the model’s performance.
- Visualized and pre-processed the highly imbalanced dataset using Python tools.

### 3. Exploratory Data Analysis (EDA)
- Conducted exploratory data analysis, including univariate and bivariate analysis, to understand the relationships and patterns in the data.

### 4. Model Selection
- Evaluated various machine learning algorithms such as Logistic Regression, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).
- Applied grid search to identify the optimal set of parameters for each model.

### 5. Model Training
- Trained the selected models using cross-validation to avoid overfitting.

### 6. Model Evaluation
- Compared models based on performance metrics such as precision, recall, F1 score, and accuracy.
- Used confusion matrix and plotted ROC curves to assess the models' performance.
- Concluded that Logistic Regression was the best classifier based on ROC curve analysis.

## Tools and Technologies Used
- **Languages**: Python
- **Libraries**: 
  - **Data Preprocessing and Visualization**: Pandas, NumPy, Matplotlib, Seaborn
  - **Machine Learning**: Scikit-Learn, XGBoost
  - **Evaluation**: ROC curve, Precision-Recall, Confusion Matrix

## Results
- **Model Performance**: Achieved a recall of up to 93% for the detection of fraudulent transactions.
- **Evaluation**: Logistic Regression outperformed other models in terms of F1 score and ROC curve analysis.

## Sample Code Snippet

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, auc

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
roc_auc = auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```

## Installation and Usage

1. **Clone the Repository**:
   ```bash
   git clone <https://github.com/priyanshu2552/Credit_Card_fraud_detection.git>
   ```

2. **Install Required Packages**:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost
   ```

3. **Run the Jupyter Notebook**:
   Open the project in Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Future Improvements
- Use advanced techniques like ensemble models or deep learning methods (e.g., neural networks) to improve fraud detection.
- Incorporate real-time data processing for real-time fraud detection.
- Apply data balancing techniques like SMOTE to handle class imbalance more effectively.

## License
This project is intended for educational purposes and personal use.
