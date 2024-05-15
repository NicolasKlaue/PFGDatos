import time
import psutil  # For monitoring system resources
import memory_profiler  # For memory profiling
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataclasses import dataclass
from transformers import pipeline
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import json

def save_resource_measurements_to_file(resource_measurements, filename='resource_measurements.json'):
    with open(filename, 'w') as f:
        json.dump(resource_measurements, f)

def measure_resources():
    start_time = time.time()  # Start time
    
    # Initialize memory profiler
    mem_profile = memory_profiler.LineProfiler()

    # Function to monitor memory usage
    @mem_profile
    def monitor_memory_usage():

        scaler = MinMaxScaler()

        #TONE CLASSIFIER
        classifierTone = joblib.load("tone_classifier_model.pkl")
        vectorizerTone = joblib.load("tfidf_vectorizer.pkl")

        classifierTopic = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

        MLPModel = load_model("ModeloMLP.keras", compile=False)

        @dataclass
        class Email():
            Subject: str
            Body: str
        label_urgency_dict = {
          "energy infrastructure": 3,
          "government & politics": 3,
          "legal affairs": 2,
          "calendar & scheduling": 1,
          "employment": 1,
          "personal & social": 0,
          "customer support": 4,
          "business document": 5,
          "energy trading": 3,
          "project management": 4,
          "meetings & events": 2,
          "project-specific": 4,
          "human resources": 3,
          "contract management": 3,
          "marketing & promotion": 2,
          "information technology": 2,
          "external affairs": 3,
          "media & press": 4,
          "finance": 5,
          "corporate governance": 5,
          "energy services": 4,
          "organization": 3,
          "utilities": 1,
          "phone communication": 2,
          "safety & emergency": 3,
          "market research": 2,
          "email management": 1,
          "education & training": 2
          }
          # Load the dataset
        data = pd.read_csv('DatasetToTestAllSystem.csv')
        def RateTone(emailInput:Email):
                    # Example email to predict
               email = """
               Subject:
               {Subject}
               Body:
               {Body}""".format(Subject = emailInput.Subject, Body = emailInput.Body)
               # Vectorize the email
               email_vectorized = vectorizerTone.transform([email])

               # Predict the tone of the email
               predicted_tone = classifierTone.predict(email_vectorized)

               toneClass = 1 if predicted_tone[0] == "formal" else -1 if predicted_tone[0] == "casual" else 0
               return toneClass

        def RateEmail(email:Email):
               sequence_to_classify = """
               Subject
               {Subject}
               Body
               {Body}""".format(Subject = email.Subject, Body = email.Body)

               classDict = classifierTopic(sequence_to_classify, list(label_urgency_dict.keys()), multi_label=True)
               filtered_predictions = [(label, score) for label, score in zip(classDict['labels'], classDict['scores']) if score > 0.3]

               sorted_predictions = sorted(filtered_predictions, key=lambda x: x[1], reverse=True)

               top_3_classifications = [label for label, _ in sorted_predictions[:3]]

               urgency_ratings = [label_urgency_dict[classification] for classification in top_3_classifications]
               if not top_3_classifications:
                    top_3_classifications = ["other"]
                    urgency_rating=0
               else:
                    urgency_rating = max(urgency_ratings)
               print("Still missing")
               return {urgency_rating}

        # Replace the values in the 'predicted_category' and 'predicted_tone' columns in the dataset
        data['predicted_category'] = data.apply(lambda row: RateEmail(Email(row['Subject'], row['Body'])), axis=1)
        data['predicted_tone'] = data.apply(lambda row: RateTone(Email(row['Subject'], row['Body'])), axis=1)

        X = data[['predicted_categories', 'predicted_tone']].values
        X = scaler.fit_transform(X)
        predictions = MLPModel.predict(X)

        predicted_urgency = np.argmax(predictions, axis=1)
        actual_urgency = data['Urgency'].values
        data['Predicted_urgency'] = predicted_urgency
        data.to_csv("FullPredictions.csv")
        conf_matrix = confusion_matrix(actual_urgency, predicted_urgency)

        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.show()

    # Start monitoring memory usage
    monitor_memory_usage()

    # Measure CPU time
    cpu_time = time.time() - start_time

    # Measure disk usage
    disk_usage = os.path.getsize('DatasetToTestAllSystem.csv')  # Example file size, replace with actual file size if needed

    # Get system resources usage
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent

    # Return resource measurements
    return {
        'cpu_time': cpu_time,
        'memory_percent': memory_percent,
        'cpu_percent': cpu_percent,
        'disk_usage': disk_usage
    }

# Call the function to measure resources
resource_measurements = measure_resources()

print("CPU Time:", resource_measurements['cpu_time'])
print("Memory Usage:", resource_measurements['memory_percent'])
print("CPU Percent:", resource_measurements['cpu_percent'])
print("Disk Usage:", resource_measurements['disk_usage'])

