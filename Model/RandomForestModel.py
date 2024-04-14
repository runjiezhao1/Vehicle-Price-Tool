import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import svm, datasets

def print_metrics(model,X_train,X_test,y_train,y_test):
    predictions_train = model.predict(X_train) 
    predictions_test = model.predict(X_test)

    final_mse_train = mean_squared_error(y_train, predictions_train)
    final_mse_test = mean_squared_error(y_test, predictions_test)

    print(f'Final RMSE on the train set: {np.sqrt(final_mse_train)}')
    print(f'Final RMSE on the test set: {np.sqrt(final_mse_test)}')

def RandomForestModel(X_base, y):
    print("enter into random forest model")
    x_train, x_test, y_train, y_test = train_test_split(X_base, y, test_size=0.3, random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [30, 40, 50],
        'min_samples_split': [8, 12],
        'min_samples_leaf': [4, 5]
    }
    parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
    svc = svm.SVC()
    model = RandomForestRegressor(n_estimators=100, 
                                      random_state=42,
                                      max_depth = 16,
                                      max_features=0.3,
                                      bootstrap=True,
                                      max_samples=0.5,
                                      ccp_alpha = 1000,
                                      n_jobs=-1) 


    model.fit(x_train, y_train)
    print_metrics(model,x_train,x_test,y_train,y_test)