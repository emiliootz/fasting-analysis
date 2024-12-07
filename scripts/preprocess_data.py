import pandas as pd
import os

# Paths
raw_data_path = os.path.join('data', 'raw', 'fasting_data1.csv')
processed_data_path = os.path.join('data', 'processed', 'fasting_data_processed.csv')

# Load the raw dataset
data = pd.read_csv(raw_data_path)

# Combine Date and Time into a single Datetime column
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], errors='coerce')

# Save the processed dataset
data.to_csv(processed_data_path, index=False)
print(f"Processed data saved to {processed_data_path}")
