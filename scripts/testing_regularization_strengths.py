# Emilio Ortiz
# testing and tuning hyperparameters by trying different regularization strengths

import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('./visualizations', exist_ok=True)
train_data = pd.read_csv('./data/train.csv')
validation_data = pd.read_csv('./data/validation.csv')

X_train = train_data[['Average Hunger', 'Average Mood', 'Average Craving', 'Difficulty Score Normalized']]
y_train = train_data['Completion']
X_val = validation_data[['Average Hunger', 'Average Mood', 'Average Craving', 'Difficulty Score Normalized']]
y_val = validation_data['Completion']

results = []
for C in np.logspace(-4, 4, 10):
    scores = cross_val_score(LogisticRegression(C=C, solver='liblinear', random_state=42), X_train, y_train, cv=5, scoring='accuracy')
    results.append((C, np.mean(scores)))

results_df = pd.DataFrame(results, columns=['C', 'Mean Accuracy'])

plt.figure(figsize=(10, 6))
plt.semilogx(results_df['C'], results_df['Mean Accuracy'], marker='o', label='Cross-Validation Accuracy', color='blue')
plt.title('Logistic Regression Tuning: Regularization Strength', fontsize=16)
plt.xlabel('C (Inverse Regularization Strength)', fontsize=14)
plt.ylabel('Mean Accuracy', fontsize=14)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plot_path = os.path.join('./visualizations', 'accuracy_vs_C.png')
plt.savefig(plot_path, dpi=300)
plt.close()

best_C = results_df.loc[results_df['Mean Accuracy'].idxmax(), 'C']
LogisticRegression(C=best_C, solver='liblinear', random_state=42).fit(X_train, y_train)
validation_accuracy = accuracy_score(y_val, LogisticRegression(C=best_C, solver='liblinear', random_state=42).predict(X_val))
