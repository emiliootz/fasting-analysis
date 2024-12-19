# Emilio Ortiz
# getting the learning curve 

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import os

data = pd.concat([pd.read_csv('./data/train.csv'), pd.read_csv('./data/validation.csv')])
X = data[['Average Hunger', 'Average Mood', 'Average Craving', 'Difficulty Score Normalized']]
y = data['Completion']
model = LogisticRegression(C=5.0, solver='liblinear', random_state=42)

train_sizes, train_scores, test_scores = learning_curve(
    model, X, y,
    cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training Cost', color='green', marker='o')
plt.plot(train_sizes, test_mean, label='Validation Cost', color='red', marker='o')
plt.title('Learning Curve: Logistic Regression', fontsize=16)
plt.xlabel('Training Set Size', fontsize=14)
plt.ylabel('Cost', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(True)
os.makedirs('./visualizations', exist_ok=True)
plt.savefig(os.path.join('./visualizations', 'learning_curve.png'), dpi=300)
plt.close()
