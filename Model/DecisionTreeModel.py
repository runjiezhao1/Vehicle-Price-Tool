import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import svm, datasets
from sklearn import tree
from sklearn.tree import export_graphviz, DecisionTreeRegressor
from subprocess import call
import matplotlib.pyplot as plt

def print_metrics(model,x_train,x_test,y_train,y_test):
    predictions_train = model.predict(x_train) 
    predictions_test = model.predict(x_test)

    final_mse_train = mean_squared_error(y_train, predictions_train)
    final_mse_test = mean_squared_error(y_test, predictions_test)

    print(f'Mean squared error of train set: {np.sqrt(final_mse_train)}')
    print(f'Mean squared error of test set: {np.sqrt(final_mse_test)}')

def DecisionTreeModel(X_base, y):
    print("enter into decision tree model")
    x_train, x_test, y_train, y_test = train_test_split(X_base, y, test_size=0.3, random_state=42)
    model = DecisionTreeRegressor(ccp_alpha=1000, random_state=42)
    model.fit(x_train, y_train)
    print_metrics(model,x_train,x_test,y_train,y_test)