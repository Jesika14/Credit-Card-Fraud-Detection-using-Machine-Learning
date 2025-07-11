# Credit Card Fraud Detection using Machine Learning

This project explores various machine learning models to detect fraudulent credit card transactions using a highly imbalanced dataset. We applied tree-based and probabilistic classifiers, evaluated them using appropriate metrics for imbalanced data, and explored the impact of oversampling techniques like SMOTE.

## Problem Statement

Detect fraudulent credit card transactions using anonymized features (V1–V28) from a real-world financial dataset. The key challenge lies in the high class imbalance between fraudulent and legitimate transactions.

---

## Dataset Summary

- 284,807 transactions
- 492 frauds (~0.17%)
- Features: 30 (anonymized PCA components, time, and amount)
- No missing values

---

## Techniques Used

### Data Exploration & Preprocessing
- Correlation matrix
- Distribution analysis
- Feature selection (none dropped as PCA features used)
- LDA attempted, but too much information loss

### Models Implemented
- Decision Tree
- Random Forest
- Bagging Classifier
- XGBoost
- Logistic Regression
- Gaussian Naive Bayes
- K-Nearest Neighbors

### Model Evaluation Metrics
Accuracy was not used due to class imbalance.

Used:
- Precision
- Recall
- F1-Score
- Area Under Precision-Recall Curve (AUPRC) – primary metric
- Confusion Matrix

---

## Oversampling (SMOTE)

Applied SMOTE (Synthetic Minority Oversampling Technique) to balance class distribution and compare model performance before and after oversampling.

---

## Results

### Top Performers on Original Data
| Model              | Recall | Precision | F1-Score | AUPRC   |
|-------------------|--------|-----------|----------|---------|
| XGBoost            | 0.85   | 0.94      | 0.89     | 0.91    |
| Voting Classifier  | 0.83   | 0.94      | 0.88     | 0.91    |
| Bagging            | 0.82   | 0.94      | 0.88     | 0.89    |

### Top Performers on Oversampled Data
| Model              | Recall | Precision | F1-Score | AUPRC   |
|-------------------|--------|-----------|----------|---------|
| Voting Classifier  | 0.86   | 0.91      | 0.88     | 0.89    |
| Random Forest      | 0.87   | 0.90      | 0.88     | 0.89    |
| XGBoost            | 0.86   | 0.89      | 0.88     | 0.89    |

---

## Conclusion

- Tree-based models outperformed other classifiers due to their ability to handle complex feature interactions.
- Oversampling improved recall and generalization slightly but did not drastically change the model ranking.
- AUPRC is the most reliable metric for such imbalanced classification tasks.

---

## Files in Repository

- `notebook.ipynb`: Main Jupyter notebook with all model code, visualizations, and evaluations
- `report.pdf`: Final PDF report summarizing the methodology and findings
- `README.md`: Project summary (this file)

---

## Technologies Used

- Python
- Scikit-learn
- XGBoost
- Matplotlib / Seaborn
- SMOTE (via imbalanced-learn)

---


