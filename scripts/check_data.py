#Emilio Ortiz
# Checking the data to make sure there are no missing values

import pandas as pd

data = pd.read_csv('./data/fastingdata.csv')

missing = data.isnull().sum()

print("Missing Values Summary:\n")
print(missing = data.isnull().sum())

if missing.sum() == 0:
    print("\nNo missing values.")
else:
    print("\nThere are missing values.")
