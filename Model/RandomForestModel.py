import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn import svm, datasets
from sklearn import tree
from sklearn.tree import export_graphviz
from subprocess import call
import matplotlib.pyplot as plt

def print_metrics(model,x_train,x_test,y_train,y_test):
    predictions_train = model.predict(x_train) 
    predictions_test = model.predict(x_test)

    final_mse_train = mean_squared_error(y_train, predictions_train)
    final_mse_test = mean_squared_error(y_test, predictions_test)

    print(f'Mean squared error of train set: {np.sqrt(final_mse_train)}')
    print(f'Mean squared error of test set: {np.sqrt(final_mse_test)}')

def RandomForestModel(X_base, y):
    print("enter into random forest model")
    x_train, x_test, y_train, y_test = train_test_split(X_base, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth = 16, max_features=0.3, bootstrap=True, max_samples=0.5, ccp_alpha = 1000, n_jobs=-1) 
    model.fit(x_train, y_train)
    print_metrics(model,x_train,x_test,y_train,y_test)
    tree.plot_tree(model.estimators_[0])
    plt.show()
    # export_graphviz(model,out_file='tree.dot')
    # call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    #Image(filename = 'tree.png')