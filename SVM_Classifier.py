import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
blink_data = pd.read_csv('eeg_data_blink.csv')
crunch_data = pd.read_csv('eeg_data_crunch.csv')
relaxed_data = pd.read_csv('eeg_data_relaxed.csv')

# Combine the datasets
data = pd.concat([blink_data, crunch_data, relaxed_data])
X = data.drop('Label', axis=1)  # Extract features
y = data['Label']  # Extract labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'eeg_classifier_model.pkl')

print("Model saved successfully.")
