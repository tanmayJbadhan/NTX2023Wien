import pandas as pd

# Load the CSV file
df = pd.read_csv('eeg_data_blink.csv')

# Replace "Crunch" with "Blink" in the 'Label' column
df['Label'] = df['Label'].replace('Crunch', 'Blink')

# Save the updated DataFrame back to the CSV file
df.to_csv('eeg_data_blink.csv', index=False)
