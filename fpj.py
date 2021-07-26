from datetime import date
from nsepy import get_history
import pandas as pd
import numpy as np
import talib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

# import Date
import pickle
import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")


pd.set_option("display.max_rows", None, "display.max_columns", None)



stock_names=['TATAMOTORS','HDFC','AXISBANK','CIPLA','DRREDDY','GLENMARK','COALINDIA']


def get_testing_data():

    stock_names=['TATAMOTORS','HDFC','AXISBANK','CIPLA','DRREDDY','GLENMARK','COALINDIA']

    for i in range(0,len(stock_names)):

        stock_name=stock_names[i]


        df = get_history(symbol=stock_name,
                           start=date(2020,8,8),
                           end=date.today())

        print(i)


        df.to_csv('prediction/'+str(stock_name)+'.csv')

# get_testing_data()



def preprocess_rf_data(df):

    zero=1
    one=1
    gap=5
    df['updown_after_gap'] = 0
    for i in range(gap, len(df) - gap):
        if (df['Close'][i - gap] > df['Close'][i]):
            df['updown_after_gap'][i - gap] = 0
            zero=zero+1
        else:
            df['updown_after_gap'][i - gap] = 1
            one=one+1
    df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
    df['stdev'] = talib.STDDEV(df['Close'], timeperiod=14, nbdev=1)

    df['trd_qty_per_trade'] = df['Volume'] / df['Trades']

    for i in range(2, 51, 7):
        # df['sma' + str(i)] = talib.SMA(df['Close'], timeperiod=i)/df['Close']
        df['sma' + str(i)] = talib.SMA(df['Close'], timeperiod=i)

    df = df[['Open', 'High', 'Low', 'Close', 'VWAP', 'Volume',
             'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble',
             'trd_qty_per_trade', 'sma2', 'sma9', 'sma16', 'sma23', 'sma30', 'sma37', 'sma44', 'atr',
             'stdev', 'updown_after_gap']]


    return df



def create_features():
    for i in range(0,len(stock_names)):


        stock_name=stock_names[i]

        df= pd.read_csv('prediction/'+str(stock_name)+'.csv')

        preprocess_rf_data(df=df)

        df.to_csv('prediction/' + str(stock_name) + '.csv')

# create_features()







