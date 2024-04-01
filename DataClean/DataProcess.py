import pandas as pd
from sklearn.preprocessing import LabelEncoder

def data_preprocess():
    pd.set_option('display.max_columns', None)
    label_encoder = LabelEncoder()
    prices = 'dataset/car_prices.csv'
    prices_df = pd.read_csv(prices)
    prices_df['make_model'] = prices_df['make'] + ' ' + prices_df['model']
    prices_df = prices_df.dropna()
    prices_df = prices_df[['year','make_model','condition','odometer','mmr','sellingprice']]
    prices_df['make_model'] = label_encoder.fit_transform(prices_df['make_model'])
    return prices_df