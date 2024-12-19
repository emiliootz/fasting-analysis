import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)
import matplotlib.pyplot as plt
import joblib

# Load the test dataset
test_file_path = 'data/test_data.csv'
test_data = pd.read_csv(test_file_path)

# Define features and target
features = ['Average Hunger', 'Average Craving', 'Average Mood']  # Update with relevant features
target_column = 'Completed Fast'  # The target variable

X_test = test_data[features]
y_test = test_data[target_column]

# Load the trained model
model_path = 'data/logistic_regression_model.pkl'
model = joblib.load(model_path)

# Standardize test data
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)

# Predict probabilities and labels
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for the positive class

# Evaluate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_prob)

print("Model Performance on Test Data:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot AUC-ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'AUC = {auc_roc:.2f}')
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.title('ROC Curve', fontsize=16)
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.legend(loc='lower right')
plt.grid(alpha=0.5)
plt.tight_layout()
plt.savefig('visualizations/roc_curve.png', dpi=300)
print("ROC Curve saved to visualizations/roc_curve.png")