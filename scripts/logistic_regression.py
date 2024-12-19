# Emilio Ortiz
# logistic regression on training+validation data and print out AUC-ROC curve

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import joblib

data = pd.concat([pd.read_csv('./data/train.csv'), pd.read_csv('./data/validation.csv')])
X_full_train = data[['Average Hunger', 'Average Mood', 'Average Craving', 'Difficulty Score Normalized']]
y_full_train = data['Completion']

test = pd.read_csv('./data/test.csv')

X = test[['Average Hunger', 'Average Mood', 'Average Craving', 'Difficulty Score Normalized']]
y = test['Completion']

model = LogisticRegression(C=5.0, solver='liblinear', random_state=42)
model.fit(X_full_train, y_full_train)
joblib.dump(model, './data/logistic_regression_model.joblib')

prediction = model.predict(X)
probability = model.predict_proba(X)[:, 1]

accuracy = accuracy_score(y, prediction)
precision = precision_score(y, prediction)
recall = recall_score(y, prediction)
auc_roc = roc_auc_score(y, probability)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test AUC-ROC: {auc_roc:.4f}")

fpr, tpr, _ = roc_curve(y, probability)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_roc:.4f}', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.title('AUC-ROC Curve', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.savefig(os.path.join('./visualizations', 'auc_roc_curve.png'), dpi=300)
plt.close()