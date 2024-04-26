import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Load the dataset from CSV
data = pd.read_csv("FinalCleanDatabase.csv")

# Drop rows with missing values
data.dropna(inplace=True)
# Find labels with only one occurrence
single_occurrence_labels = data['predicted_tone'].value_counts()[data['predicted_tone'].value_counts() == 1].index

# Filter out samples with labels that have only one occurrence
data_filtered = data[~data['predicted_tone'].isin(single_occurrence_labels)]

# Concatenate "Subject" and "Body" into a single column "Email"
data_filtered['Email'] = data_filtered['Subject'] + " " + data_filtered['Body']

# Drop "Subject" and "Body" columns
data_filtered.drop(columns=['Subject', 'Body'], inplace=True)

# Select relevant columns
data_filtered = data_filtered[["Email", "predicted_tone"]]

# Split the dataset into features (X) and target (y)
X = data_filtered["Email"]
y = data_filtered["predicted_tone"]
# Split the dataset into training and testing sets with balanced class distribution
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=16)


# Create a TF-IDF vectorizer to convert emails into numerical feature vectors
vectorizer = TfidfVectorizer()

# Fit and transform the training data with the vectorizer
X_train_vectorized = vectorizer.fit_transform(X_train)
# Calculate class weights

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Convert class weights to a dictionary
class_weight_dict = dict(zip(np.unique(y_train), class_weights))

# Define the classifier (Support Vector Machine) with class weights
classifier = make_pipeline(StandardScaler(with_mean=False), SVC(kernel='linear', class_weight=class_weight_dict))

# Fit the classifier on the training data
classifier.fit(X_train_vectorized, y_train)

# Fit the classifier on the training data
classifier.fit(X_train_vectorized, y_train)

# Predict on the testing set
X_test_vectorized = vectorizer.transform(X_test)
y_pred = classifier.predict(X_test_vectorized)
train_y_pred = classifier.predict(X_train_vectorized)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
print("Classification Report:")
# Generate classification report
classification_rep = classification_report(y_test, y_pred, zero_division=1)
train_classification_rep = classification_report(train_y_pred, y_train, zero_division=1)
print(classification_rep)
print("/////////////////////////////")
print(train_classification_rep)

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
train_conf_matrix = confusion_matrix(y_train, train_y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot and save the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()
# Plot and save the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(train_conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.savefig("train_confusion_matrix.png")
plt.show()

# Save the trained model and vectorizer to files
joblib.dump(classifier, "tone_classifier_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

# Save metrics to a text file
with open("metrics.txt", "w") as f:
    f.write("Accuracy: {}\n".format(accuracy))
    f.write("\nTest Classification Report:\n")
    f.write(classification_rep)
    f.write("\nTrain Classification Report:\n")
    f.write(train_classification_rep)

print("Model trained and saved successfully!")
