"""
@Author: Tomasz Gruca
@Student ID: 21733075
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, make_scorer

# Load the dataset
data = pd.read_csv("steel.csv")

# Define the independent and dependent variables
independent_variables = ['normalising_temperature', 'tempering_temperature', 'percent_silicon', 'percent_chromium', 'percent_copper', 'percent_nickel', 'percent_sulphur', 'percent_carbon', 'percent_manganese']
dependent_variable = 'tensile_strength'

# Split the data into independent and dependent variables
x = data.loc[:, independent_variables]
y = data.loc[:, dependent_variable]


kf = KFold(n_splits=10, shuffle=True, random_state=0)

# Create a Support Vector Regressor model
svr = SVR()

# Create a Random Forest Regressor model
rf = RandomForestRegressor(random_state=0)

# Define the parameters for the grid search
scoring = { 'MSE': make_scorer(mean_squared_error), 
           'MAE': make_scorer(mean_absolute_error), 
           'MAPE': make_scorer(mean_absolute_percentage_error)
             }

svr_score_mse = cross_val_score(svr, x, y, cv=kf, scoring=make_scorer(mean_squared_error))
svr_score_mae = cross_val_score(svr, x, y, cv=kf, scoring=make_scorer(mean_absolute_error))
svr_score_mape = cross_val_score(svr, x, y, cv=kf, scoring=make_scorer(mean_absolute_percentage_error))

rf_score_mse = cross_val_score(rf, x, y, cv=kf, scoring=make_scorer(mean_squared_error))
rf_score_mae = cross_val_score(rf, x, y, cv=kf, scoring=make_scorer(mean_absolute_error))
rf_score_mape = cross_val_score(rf, x, y, cv=kf, scoring=make_scorer(mean_absolute_percentage_error))

print("Default Support Vector Regressor Model") 
print("MSE: ", svr_score_mse.mean())
print("MAE: ", svr_score_mae.mean())
print("MAPE: ", svr_score_mape.mean())

print("\nDefault Random Forest Regressor Model")
print("MSE: ", rf_score_mse.mean())
print("MAE: ", rf_score_mae.mean())
print("MAPE: ", rf_score_mape.mean())

# Tune the SVR model
svr_param_grid = {'kernel': ['linear', 'rbf', 'poly'], 'epsilon': [0.1, 0.2, 0.5]}
svr_grid = GridSearchCV(estimator=svr, param_grid=svr_param_grid, cv=kf, scoring=scoring, refit='MSE', n_jobs=-1)
svr_grid.fit(x, y)

print("\nTuned SVR Model")
print("Best Parameters: ", svr_grid.best_params_)
print("Average MSE: ", svr_grid.best_score_)
print("Average MAE: ", -cross_val_score(svr_grid.best_estimator_, x, y, cv=kf, scoring='neg_mean_absolute_error').mean())
print("Average MAPE: ", -cross_val_score(svr_grid.best_estimator_, x, y, cv=kf, scoring='neg_mean_absolute_percentage_error').mean())

# Tune the Random Forest model
rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, 30]}
rf_grid = GridSearchCV(rf, rf_param_grid, cv=kf,  scoring=scoring, refit='MSE', n_jobs=-1)
rf_grid.fit(x, y)

print("\nTuned Random Forest Model")
print("Best Parameters: ", rf_grid.best_params_)
print("Average MSE: ", rf_grid.best_score_)
print("Average MAE: ", -cross_val_score(rf_grid.best_estimator_, x, y, cv=kf, scoring='neg_mean_absolute_error').mean())
print("Average MAPE: ", -cross_val_score(rf_grid.best_estimator_, x, y, cv=kf, scoring='neg_mean_absolute_percentage_error').mean())

# Plotting the results
svr_default_mse = svr_score_mse.mean()
svr_tuned_mse = svr_grid.best_score_
svr_default_mape = svr_score_mape.mean()
svr_tuned_mape = -cross_val_score(svr_grid.best_estimator_, x, y, cv=kf, scoring='neg_mean_absolute_percentage_error').mean()

rf_default_mse = rf_score_mse.mean()
rf_tuned_mse = rf_grid.best_score_
rf_default_mape = rf_score_mape.mean()
rf_tuned_mape = -cross_val_score(rf_grid.best_estimator_, x, y, cv=kf, scoring='neg_mean_absolute_percentage_error').mean()

# Convert negative scores to positive
svr_tuned_mape = abs(svr_tuned_mape)
rf_tuned_mape = abs(rf_tuned_mape)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

x = ['Default', 'Tuned']

# Plot for SVR MSE
axes[0, 0].bar(x, [svr_default_mse, svr_tuned_mse], color=['blue', 'orange'])
axes[0, 0].set_title('SVR MSE')
axes[0, 0].set_ylabel('MSE')
axes[0, 0].set_ylim([0, max(svr_default_mse, svr_tuned_mse) * 1.1])

# Plot for SVR MAPE
axes[0, 1].bar(x, [svr_default_mape, svr_tuned_mape], color=['blue', 'orange'])
axes[0, 1].set_title('SVR MAPE')
axes[0, 1].set_ylabel('MAPE')
axes[0, 1].set_ylim([0, max(svr_default_mape, svr_tuned_mape) * 1.1])

# Plot for Random Forest MSE
axes[1, 0].bar(x, [rf_default_mse, rf_tuned_mse], color=['blue', 'orange'])
axes[1, 0].set_title('Random Forest MSE')
axes[1, 0].set_ylabel('MSE')
axes[1, 0].set_ylim([0, max(rf_default_mse, rf_tuned_mse) * 1.1])

# Plot for Random Forest MAPE
axes[1, 1].bar(x, [rf_default_mape, rf_tuned_mape], color=['blue', 'orange'])
axes[1, 1].set_title('Random Forest MAPE')
axes[1, 1].set_ylabel('MAPE')
axes[1, 1].set_ylim([0, max(rf_default_mape, rf_tuned_mape) * 1.1])

plt.tight_layout()
plt.show()