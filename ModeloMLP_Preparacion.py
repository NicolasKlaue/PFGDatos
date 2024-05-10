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

data['predicted_tone'] = data['predicted_tone'].apply(lambda x: 1 if x == 'formal' else -1 if x == 'casual' else 0)

print(data['predicted_tone'].unique())

label_dict = {
"energy infrastructure" : 3,
"government & politics" : 3,
"legal affairs" : 2,
"calendar & scheduling" : 3,
"employment" : 2,
"personal & social" : 1,
"customer support" : 4,
"business document" : 5,
"energy trading" : 3,
"project management" : 4,
"meetings & events" : 2,
"project-specific" : 4,
"human resources" : 3,
"contract management" : 3,
"marketing & promotion" : 2,
"information technology" : 2,
"external affairs" : 3,
"media & press" : 4,
"finance" : 5,
"corporate governance" : 5,
"energy services" : 4,
"organization" : 3,
"utilities" : 1,
"phone communication" : 2,
"safety & emergency" : 3,
"market research" : 2,
"email management" : 1,
"education & training" : 2,
"other": 0
}
print(data["predicted_categories"].unique())
data = data.dropna()
data = data.drop(data[data["predicted_categories"] == "other"].index)
data['predicted_categories'] = data['predicted_categories'].apply(lambda x: label_dict[x] if x in label_dict.keys() else 0)
print(data["predicted_categories"].value_counts())
data = data.drop(columns=["Subject", "Body"])
data.to_csv("DatasetToTrainMLP.csv")