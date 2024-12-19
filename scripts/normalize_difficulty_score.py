# Emilio Ortiz
# normalize the difficulty score min/max X

import pandas as pd

data = pd.read_csv('./data/engineered_fastingdata.csv')

minX = data['Difficulty Score'].min()
maxX = data['Difficulty Score'].max()

data['Difficulty Score Normalized'] = ((data['Difficulty Score'] - minX) / (maxX - minX))

data.to_csv('./data/engineered_fastingdata_normalized.csv', index=False)
