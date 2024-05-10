from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import l2
import seaborn as sns
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
tf.random.set_seed = 16
# Load the dataset
data = pd.read_csv('DatasetToTrainMLP.csv')

# Split the dataset into input and output variables
X = data[['predicted_categories', 'predicted_tone']].values
y = data['Urgency'].values

# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to the data and transform it
X = scaler.fit_transform(X)
y = np.array(y).reshape(-1, 1)  # Reshape y to make it compatible with regression

# Compute class weights
class_labels = np.unique(y)
class_weights = compute_class_weight('balanced', classes=class_labels, y=y.flatten())

# Convert class weights to a dictionary
class_weight_dict = dict(zip(class_labels, class_weights))

# Define a custom loss function with class weights
def weighted_mse(class_weights):
    def loss(y_true, y_pred):
        weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
        return tf.reduce_mean(tf.square(y_true - y_pred) * weights)
    return loss

# Create the MLP model with the custom loss function
model = Sequential()
model.add(Dense(16, input_dim=2, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model with the custom loss function
model.compile(loss=weighted_mse(class_weights), optimizer='adam', metrics=['mean_squared_error'])

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=16)

# Train the model with the class weights and validation data
history = model.fit(X_train, y_train, epochs=20, batch_size=32, class_weight=class_weight_dict, validation_data=(X_test, y_test))

# Evaluate the model
loss, mse = model.evaluate(X_test, y_test)
print(f"Mean Squared Error: {mse}")

# Plot the training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the predicted output
predictions = model.predict(X_test)
# Round the predicted values to the nearest integer
rounded_predictions = np.round(predictions).astype(int)

# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, rounded_predictions)

# Calculate confusion matrix percentages
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

# Plot the confusion matrix as percentages
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt='.2f', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix (Percentages)')
plt.show()

model.save("ModeloMLP.h5")
