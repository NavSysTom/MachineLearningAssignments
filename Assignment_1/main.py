"""
@Author: Tomasz Gruca
@Student ID: 21733075
"""

import pandas as pd
from matplotlib import pyplot as plt

training_file = "Datasets/wildfires_training.csv"
testing_file = "Datasets/wildfires_testing.csv"

independent_variables = ['year', 'temp', 'humidity', 'rainfall', 'drought_code', 'buildup_index', 'day', 'month', 'wind_speed']
dependent_variable = 'fire'

fire_training = pd.read_csv(training_file)
print(fire_training.head())
