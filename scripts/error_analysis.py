# Emilio Ortiz
# Analyze the test set predictions and identify common misclassifications

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from joblib import load

data = pd.read_csv('./data/test.csv')
X = data[['Average Hunger', 'Average Mood', 'Average Craving', 'Difficulty Score Normalized']]
y = data['Completion']
model = load('./data/logistic_regression_model.joblib')
prediction = model.predict(X)

print(f"Confusion Matrix:\n {confusion_matrix(y, prediction)}")
print(f"\nClassification Report:\n{classification_report(y, prediction, target_names=['Not Completed', 'Completed'])}")

false_positives = data[(y == 0) & (prediction == 1)]
false_negatives = data[(y == 1) & (prediction == 0)]

print(f"False Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")
