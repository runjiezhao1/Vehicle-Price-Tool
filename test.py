import DataClean.DataProcess as dp
import Model.LinearModel as LM
import Model.RandomForestModel as RF
import numpy as np


df = dp.data_preprocess()
X = df[['year','make_model','condition','odometer','mmr']]
X_base = X.loc[:,:].values
y = df['sellingprice'].values
degrees = np.arange(1, 20)

# print(df[df['mmr'] > 0])
# mse = abs(df['mmr'] - df['sellingprice']).sum() / df.shape[0]
# print(mse)
#LM.LinearRegressionModel(X_base,y,degrees)
RF.RandomForestModel(X_base, y)