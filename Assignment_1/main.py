"""
@Author: Tomasz Gruca
@Student ID: 21733075
"""

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression 

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


# Train the model from randomstate=0 to max iterations=1000 to train the logistic regression model
clf = LogisticRegression(random_state=0, max_iter=1000).fit(x_training, y_training)

# Preparing to test with new data
x_testing = fire_testing.loc[:,independent_variables]
y_testing = fire_testing.loc[:,dependent_variable]

# Predict the testing data
y_pred = clf.predict(x_testing)

# Print the predicted values
accuracy_training_data= clf.score(x_training, y_training)
accuracy_test_data = clf.score(x_testing, y_testing)
print("\nDefault Logistic Regression Model")
print("Accuracy of Training Data: ", accuracy_training_data)
print("Accuracy of New Data: ", accuracy_test_data)

clf_tuned = LogisticRegression(random_state=0, max_iter=2000, C=0.8).fit(x_training, y_training)

# Predict the testing data
accuracy_training_tuned_data = clf_tuned.score(x_training, y_training)
accuracy_test_data_tuned = clf_tuned.score(x_testing, y_testing)

print("\nTuned Logistic Regression Model")
print("Accuracy of Training Data: ", accuracy_training_tuned_data)
print("Accuracy of New Data: ", accuracy_test_data_tuned)

clf_tuned2 = LogisticRegression(random_state=0, max_iter=5000, C=1.5).fit(x_training, y_training)

# Predict the testing data
accuracy_training_tuned_data2 = clf_tuned2.score(x_training, y_training)
accuracy_test_data_tuned2 = clf_tuned2.score(x_testing, y_testing)

print("\nTuned Logistic Regression Model 2")
print("Accuracy of Training Data: ", accuracy_training_tuned_data2)
print("Accuracy of New Data: ", accuracy_test_data_tuned2)
