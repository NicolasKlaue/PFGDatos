import joblib

# Load the trained model and TF-IDF vectorizer
classifier = joblib.load("tone_classifier_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Example email to predict
email = "Subject:\nThis is unaccpetable, the way I have been treated, I will leave this company today\nBody: There is no way of making me stay "

# Vectorize the email
email_vectorized = vectorizer.transform([email])

# Predict the tone of the email
predicted_tone = classifier.predict(email_vectorized)

print("Predicted tone:", predicted_tone[0])
