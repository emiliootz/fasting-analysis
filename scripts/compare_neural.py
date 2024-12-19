# Emilio Ortiz
# Compare Logistic Regression and Neural Network using Venn Diagram

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from matplotlib_venn import venn2

test = pd.read_csv('./data/test.csv')
training = pd.concat([pd.read_csv('./data/train.csv'), pd.read_csv('./data/validation.csv')])

Xtrain = training[['Average Hunger', 'Average Mood', 'Average Craving', 'Difficulty Score Normalized']]
ytrain = training['Completion']
Xtest = test[['Average Hunger', 'Average Mood', 'Average Craving', 'Difficulty Score Normalized']]
ytest = test['Completion']

models = {
    "Logistic Regression": LogisticRegression(C=5.0, solver='liblinear', random_state=42),
    "Neural Network": MLPClassifier(max_iter=500, random_state=42)
}
indices = {}
for name, model in models.items():
    print(f"{name}\n")
    model.fit(Xtrain, ytrain)
    prediction = model.predict(Xtest)
    misclassified = set(test.index[ytest != prediction])
    indices[name] = misclassified
    print(f"\nClassification Report:\n {classification_report(ytest, prediction)}")

plt.figure(figsize=(8, 6))
venn2(
    [indices["Logistic Regression"], indices["Neural Network"]],
    set_labels=("Logistic Regression", "Neural Network")
)
plt.title("Misclassifications", fontsize=16)
plt.savefig(os.path.join('./visualizations', 'venn_diagram.png'), dpi=300)
plt.close()
