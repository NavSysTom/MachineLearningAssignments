"""
@Author: Tomasz Gruca
@Student ID: 21733075
"""

import pandas as pd
from matplotlib import pyplot as plt

training_file = "Datasets/wildfires_training.csv"
testing_file = "Datasets/wildfires_test.csv"

independent_variables = ['year', 'temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'day', 'month', 'wind_speed']
dependent_variable = 'fire'

fire_training = pd.read_csv(training_file)
print("Training data:")
print(fire_training.head())
print(fire_training.shape)

X_train = fire_training.loc[:,independent_variables]
print("X_train:")
print(X_train.head())
print(X_train.shape)

y_training = fire_training.loc[:,dependent_variable]
print("y_training:")
print(y_training.head())
print(y_training.shape)

fire_testing = pd.read_csv(testing_file)
print("Testing data:")
print(fire_testing.head())
print(fire_testing.shape)