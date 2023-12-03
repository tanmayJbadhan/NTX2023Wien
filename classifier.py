import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from CSV files
blink_data = pd.read_csv('eeg_data_blink.csv')
print(len(blink_data))
crunch_data = pd.read_csv('eeg_data_crunch.csv')

# Concatenate the data from both files
combined_data = pd.concat([blink_data, crunch_data], ignore_index=True)
print(len(combined_data))

# Split the data into features (X) and labels (y)
X = combined_data.drop(columns=['Label'])
y = combined_data['Label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
