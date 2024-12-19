# Emilio Ortiz
# Enginner some features like completion, combining datetime, and averages

import pandas as pd
import numpy as np

fasting_data = pd.read_csv('./data/fastingdata.csv')

# make datetime column
fasting_data['Datetime'] = pd.to_datetime(fasting_data['Date'] + ' ' + fasting_data['Time'], format='%m/%d/%Y %I:%M:%S %p')

# sort by fullname and datetime and pair the fasts
fasting_data = fasting_data.sort_values(by=['Full Name', 'Datetime']).reset_index(drop=True)

start_fasts = fasting_data[fasting_data['Start or Break?'] == 'Start Fast'].reset_index(drop=True)
break_fasts = fasting_data[fasting_data['Start or Break?'] == 'Break Fast'].reset_index(drop=True)

paired_fasts = []
for name in fasting_data['Full Name'].unique():
    starts = start_fasts[start_fasts['Full Name'] == name]
    breaks = break_fasts[break_fasts['Full Name'] == name]
    
    i, j = 0, 0
    while i < len(starts) and j < len(breaks):
        start = starts.iloc[i]
        end = breaks.iloc[j]
        if end['Datetime'] > start['Datetime']:
            paired_fasts.append({
                'Full Name': name,
                'Start Time': start['Datetime'],
                'End Time': end['Datetime'],
                'Fasting Length': (end['Datetime'] - start['Datetime']).total_seconds() / 3600,
                'Hunger Level Start': start['Hunger Level'],
                'Mood Start': start['Mood'],
                'Craving Intensity Start': start['Craving Intensity'],
                'Hunger Level End': end['Hunger Level'],
                'Mood End': end['Mood'],
                'Craving Intensity End': end['Craving Intensity']
            })
            i += 1
            j += 1
        else:
            j += 1

paired_fasts = pd.DataFrame(paired_fasts)

# engineer the completion column
paired_fasts['Completion'] = paired_fasts['Fasting Length'].apply(lambda x: 1 if x >= 24 else 0)

# engineer time of day column
def categorize_time_of_day(hour):
    if 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    elif 18 <= hour < 24:
        return 'evening'
    else:
        return 'night'

paired_fasts['Start Time of Day'] = paired_fasts['Start Time'].dt.hour.apply(categorize_time_of_day)
paired_fasts['Start Day of Week'] = paired_fasts['Start Time'].dt.day_name()

# averages
paired_fasts['Average Hunger'] = (paired_fasts['Hunger Level Start'] + paired_fasts['Hunger Level End']) / 2
paired_fasts['Average Mood'] = (paired_fasts['Mood Start'] + paired_fasts['Mood End']) / 2
paired_fasts['Average Craving'] = (paired_fasts['Craving Intensity Start'] + paired_fasts['Craving Intensity End']) / 2

# interactions
paired_fasts['Hunger x Cravings'] = paired_fasts['Average Hunger'] * paired_fasts['Average Craving']
paired_fasts['Mood x Cravings'] = paired_fasts['Average Mood'] * paired_fasts['Average Craving']
paired_fasts['Craving-to-Mood Ratio'] = paired_fasts['Average Craving'] / (paired_fasts['Average Mood'] + 1e-9)
paired_fasts['Hunger-to-Craving Ratio'] = paired_fasts['Average Hunger'] / (paired_fasts['Average Craving'] + 1e-9)

# completion weight checked by oprimize_alpha script
alpha = 0.1

# engineer dificulty score
paired_fasts['Difficulty Score'] = (
    (paired_fasts['Average Hunger'] + paired_fasts['Average Craving']) - 
    (paired_fasts['Average Mood'] + alpha * paired_fasts['Completion'])
)

paired_fasts.to_csv('./data/engineered_fastingdata.csv', index=False)
