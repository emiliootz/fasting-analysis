# Emilio Ortiz
# analyzing the fasting data with the engineered features and normalized difficulty score

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

paired_fasts = pd.read_csv('./data/engineered_fastingdata_normalized.csv')

path = './visualizations'
os.makedirs(path, exist_ok=True)

sns.set_theme(style="whitegrid", context="talk")
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
sns.histplot(paired_fasts['Average Hunger'], bins=20, kde=True, ax=axes[0], color='blue')
axes[0].set_title('Distribution of Average Hunger', fontsize=16)
axes[0].set_xlabel('Hunger Level', fontsize=14)
axes[0].set_ylabel('Frequency', fontsize=14)
sns.histplot(paired_fasts['Average Mood'], bins=20, kde=True, ax=axes[1], color='green')
axes[1].set_title('Distribution of Average Mood', fontsize=16)
axes[1].set_xlabel('Mood Level', fontsize=14)
sns.histplot(paired_fasts['Average Craving'], bins=20, kde=True, ax=axes[2], color='orange')
axes[2].set_title('Distribution of Average Craving', fontsize=16)
axes[2].set_xlabel('Craving Intensity', fontsize=14)

plt.tight_layout()
plt.savefig(os.path.join(path, 'histograms.png'), dpi=300)
plt.close()

plt.figure(figsize=(12, 8))
sns.boxplot(data=paired_fasts[['Average Hunger', 'Average Mood', 'Average Craving']], palette="Set2")
plt.title('Feature Variability: Hunger, Mood, Cravings', fontsize=18)
plt.ylabel('Values', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig(os.path.join(path, 'boxplot.png'), dpi=300)
plt.close()

plt.figure(figsize=(12, 10))
correlation_matrix = paired_fasts[['Average Hunger', 'Average Mood', 'Average Craving', 'Difficulty Score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, linewidths=.5, cbar_kws={"shrink": 0.8})
plt.title('Correlation Heatmap: Features and Difficulty Score', fontsize=18)
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12, rotation=0)

plt.savefig(os.path.join(path, 'heatmap.png'), dpi=300)
plt.close()
