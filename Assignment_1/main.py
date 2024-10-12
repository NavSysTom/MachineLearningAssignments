"""
@Author: Tomasz Gruca
@Student ID: 21733075
"""

from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import seaborn as sns

# Load the iris dataset
training_file = "Datasets/wildfires_training.csv"
testing_file = "Datasets/wildfires_test.csv"

# Define the independent and dependent variables
independent_variables = ['year', 'temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'day', 'month', 'wind_speed']
dependent_variable = 'fire'

# Load the data
fire_training = pd.read_csv(training_file)

# Split the data into independent and dependent variables
x_training = fire_training.loc[:,independent_variables]
y_training = fire_training.loc[:,dependent_variable]

# Load the testing data
fire_testing = pd.read_csv(testing_file)

# Preparing to test with new data
x_testing = fire_testing.loc[:,independent_variables]
y_testing = fire_testing.loc[:,dependent_variable]

# Create a grid of scatter plots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Scatter plot for temperature
axs[0].scatter(x_testing['temp'], y_testing, label='Temperature', alpha=0.5)
axs[0].set_xlabel('Temperature')
axs[0].set_ylabel('Dependent Variable (fire)')
axs[0].set_title('Scatter Plot of Temperature vs Fire')
axs[0].legend()

# Scatter plot for rainfall
axs[1].scatter(x_testing['rainfall'], y_testing, label='Rainfall', alpha=0.5)
axs[1].set_xlabel('Rainfall')
axs[1].set_ylabel('Dependent Variable (fire)')
axs[1].set_title('Scatter Plot of Rainfall vs Fire')
axs[1].legend()

plt.tight_layout()
plt.show()

# Train the model from randomstate=0 to max iterations=1000 to train the logistic regression model
clf = LogisticRegression(max_iter=1000).fit(x_training, y_training)

# Predict the training data
accuracy_training_data= clf.score(x_training, y_training)
print("\nAccuracy of Training Data without hyperparam changes: ", accuracy_training_data)

# Predict the testing data
accuracy_test_data = clf.score(x_testing, y_testing)
print("Accuracy of Test Data without hyperparam changes: ", accuracy_test_data)


# Tuning the model with C=0.5 and max_iter=2000
clf_tuned = LogisticRegression(max_iter=2000, C=0.5).fit(x_training, y_training)

# Predict the testing data
accuracy_training_tuned_data = clf_tuned.score(x_training, y_training)
accuracy_test_data_tuned = clf_tuned.score(x_testing, y_testing)

print("\nTuned Logistic Regression Model")
print("Accuracy of Training Data with C=0.5 and max iteration of 2000: ", accuracy_training_tuned_data)
print("Accuracy of New Data with C=0.5 and max iteration of 2000: ", accuracy_test_data_tuned)

# Tuning the model with C=1.5 and max_iter=5000
clf_tuned2 = LogisticRegression(max_iter=5000, C=1.5).fit(x_training, y_training)

# Predict the testing data
accuracy_training_tuned_data2 = clf_tuned2.score(x_training, y_training)
accuracy_test_data_tuned2 = clf_tuned2.score(x_testing, y_testing)

print("\nTuned Logistic Regression Model 2")
print("Accuracy of Training Data with C=1.5 and max iteration of 5000: ", accuracy_training_tuned_data2)
print("Accuracy of New Data with C=1.5 and max iteration of 5000: ", accuracy_test_data_tuned2)

# Tuning the model with C=10 and max_iter=10000
clf_tuned3 = LogisticRegression(max_iter=10000, C=10).fit(x_training, y_training)

# Predict the testing data
accuracy_training_tuned_data3 = clf_tuned3.score(x_training, y_training)
accuracy_test_data_tuned3 = clf_tuned3.score(x_testing, y_testing)

print("\nTuned Logistic Regression Model 3")
print("Accuracy of Training Data with C=10 and max iteration of 10000: ", accuracy_training_tuned_data3)
print("Accuracy of New Data with C=10 and max iteration of 10000: ", accuracy_test_data_tuned3)

# Predict the testing data
y_pred = clf.predict(x_testing)

# Generate the confusion matrix
conf_matrix = confusion_matrix(y_testing, y_pred)

print("\nConfusion Matrix for Logistic Regression Model:")
print(conf_matrix)

# Plotting the confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fire', 'Fire'], yticklabels=['No Fire', 'Fire'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# SVM Model 
svm_clf = SVC().fit(x_training, y_training)
print("\nSVM Model Testing")

# Predict the training data
accuracy_training_svm = svm_clf.score(x_training, y_training)
print("Accuracy of Training Data without hyperparam changes: ", accuracy_training_svm)

# Predict the testing data
accuracy_test_svm = svm_clf.score(x_testing, y_testing)
print("Accuracy of Test Data without hyperparam changes: ", accuracy_test_svm)

# Tuning the model with C=0.5 
svm_clf_tuned = SVC(C=0.5).fit(x_training, y_training)

# Predict the testing data
accuracy_training_svm_tuned = svm_clf_tuned.score(x_training, y_training)
accuracy_test_svm_tuned = svm_clf_tuned.score(x_testing, y_testing)

print("\nTuned SVM Model")
print("Accuracy of Training Data with C=0.5: ", accuracy_training_svm_tuned)
print("Accuracy of New Data with C=0.5: ", accuracy_test_svm_tuned)

# Tuning the model with C=1.5
svm_clf_tuned2 = SVC(C=1.5).fit(x_training, y_training)

# Predict the testing data
accuracy_training_svm_tuned2 = svm_clf_tuned2.score(x_training, y_training)
accuracy_test_svm_tuned2 = svm_clf_tuned2.score(x_testing, y_testing)

print("\nTuned SVM Model 2")
print("Accuracy of Training Data with C=1.5: ", accuracy_training_svm_tuned2)
print("Accuracy of New Data with C=1.5: ", accuracy_test_svm_tuned2)

# Tuning the model with C=1 and kernel='linear'
svm_clf_tuned4 = SVC(C=1, kernel='linear').fit(x_training, y_training)

# Predict the testing data
accuracy_training_svm_tuned3 = svm_clf_tuned4.score(x_training, y_training)
accuracy_test_svm_tuned3 = svm_clf_tuned4.score(x_testing, y_testing)

print("\nTuned SVM Model 3")
print("Accuracy of Training Data with C=1 and kernel='linear': ", accuracy_training_svm_tuned3)
print("Accuracy of New Data with C=1 and kernel='linear': ", accuracy_test_svm_tuned3)

# Tuning the model with C=1 and kernel='poly'
svm_clf_tuned4 = SVC(C=1, kernel='poly').fit(x_training, y_training)

# Predict the testing data
accuracy_training_svm_tuned4 = svm_clf_tuned4.score(x_training, y_training)
accuracy_test_svm_tuned4 = svm_clf_tuned4.score(x_testing, y_testing)

print("\nTuned SVM Model 5")
print("Accuracy of Training Data with C=1 and kernel='poly': ", accuracy_training_svm_tuned4)
print("Accuracy of New Data with C=1 and kernel='poly': ", accuracy_test_svm_tuned4)

