import pandas as pd
import sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LinearRegressionSklearn


def LinearRegressionModel(X_base, y, degrees):

    loss_train_list = []
    loss_test_list = []

    # X_base = poly_reg_df.iloc[:, :-1].values
    # y = poly_reg_df.iloc[:, -1].values

    for d in degrees:
        poly = PolynomialFeatures(d)
        X_poly = poly.fit_transform(X_base)
        x_train, x_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.3, random_state=42)
        sds = StandardScaler()
        x_train = sds.fit_transform(x_train)
        x_test = sds.fit_transform(x_test)
        lr_model = LinearRegressionSklearn()
        lr_model.fit(x_train,y_train)
        print(lr_model.score(x_train,y_train))
        y_predict_train = lr_model.predict(x_train)
        y_predict_test = lr_model.predict(x_test)
        loss_train = sklearn.metrics.mean_squared_error(y_predict_train, y_train)
        loss_test = sklearn.metrics.mean_squared_error(y_predict_test, y_test)
        loss_train_list.append(loss_train)
        
        diffList = abs(y_predict_test - y_test)
        falseCount = 0
        for i in diffList:
            if i > 2000:
                falseCount += 1.0
        print('false rate is %f' % (falseCount / len(diffList)))

        loss_test_list.append(loss_test)
        print('degree %d, train loss %d, test loss %d' % (d, loss_train, loss_test))

    return loss_train_list, loss_test_list