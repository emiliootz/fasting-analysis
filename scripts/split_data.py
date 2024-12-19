# Emilio Ortiz
# split data into training, validation, and test

import pandas as pd
from sklearn.model_selection import train_test_split

train, temp = train_test_split(pd.read_csv('./data/engineered_fastingdata_normalized.csv'), test_size=0.4, random_state=42, shuffle=True)
validation, test = train_test_split(temp, test_size=0.5, random_state=42, shuffle=True)

train.to_csv('./data/train.csv', index=False)
validation.to_csv('./data/validation.csv', index=False)
test.to_csv('./data/test.csv', index=False)