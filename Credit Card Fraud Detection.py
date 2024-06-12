import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('C:/Users/Hitesh/Downloads/creditcard.csv')

# Data preprocessing
# (Assuming the necessary preprocessing steps are included in the script)
data = data.dropna()  # Example preprocessing step

# Define features and labels
X = data.drop('Class', axis=1)  # 'Class' is typically the label column in fraud detection datasets
Y = data['Class']

print(data.groupby('Class').mean())
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict and evaluate the model
train_predictions = model.predict(x_train)
train_accuracy = accuracy_score(y_train, train_predictions)
print("Training Accuracy:", train_accuracy)

test_predictions = model.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Test Accuracy:", test_accuracy)

# Confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, test_predictions)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_test, test_predictions)
print("Classification Report:\n", class_report)
