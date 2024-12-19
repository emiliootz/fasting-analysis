# Emilio Ortiz
# trying to optimaize the alpha used to calculate the difficulty score in feature_engineering.py 

import numpy as np
import pandas as pd

def calculate_difficulty_score(data, alpha):
    return (
        (data['Average Hunger'] + data['Average Craving']) - 
        (data['Average Mood'] + alpha * data['Completion'])
    )

def optimize_alpha(data, alpha_range):
    results = []
    for alpha in alpha_range:
        data['Difficulty Score'] = calculate_difficulty_score(data, alpha)
        correlation = data['Difficulty Score'].corr(data['Completion'])
        results.append((alpha, correlation))

    results_df = pd.DataFrame(results, columns=['Alpha', 'Correlation'])
    return results_df

# Optimize
alpha_results = optimize_alpha(pd.read_csv('./data/engineered_fastingdata.csv'), np.linspace(0.1, 10, 100))

# Best Alpha has the highest correlation
best_alpha = alpha_results.loc[alpha_results['Correlation'].idxmax(), 'Alpha']

# Print results to console
print(f"Best Alpha: {best_alpha}")
print("\nResults:")
print(alpha_results)
