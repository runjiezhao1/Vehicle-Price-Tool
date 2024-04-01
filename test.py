import DataClean.DataProcess as dp
import Model.LinearModel as LM
import numpy as np

df = dp.data_preprocess()
X = df[['year','make_model','condition','odometer','mmr']]
X_base = X.loc[:,:].values
y = df['sellingprice'].values
degrees = np.arange(1, 6)

LM.LinearRegressionModel(X_base,y,degrees)