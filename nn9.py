# successfull
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
# from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import talib



pd.set_option("display.max_rows", None, "display.max_columns", None)

# import seaborn as sns

# from datetime import datetime


# df = pd.read_csv('data/GE.csv')

# df=pd.read_csv('data1/stock_eq_fut.csv',index_col='Date')

# df=pd.read_csv('data1/EICHERMOT.csv',index_col='Date')
# df=pd.read_csv('data1/BAJAJ-AUTO.csv',index_col='Date')
# df=pd.read_csv('data1/stock_eq_fut.csv',index_col='Date')
# df=pd.read_csv('data1/GAIL.csv',index_col='Date')
# df=pd.read_csv('data1/CIPLA.csv',index_col='Date')
# df=pd.read_csv('data1/AXISBANK.csv',index_col='Date')

# df=pd.read_csv('data1/BHEL.csv',index_col='Date')
# df=pd.read_csv('data1/COALINDIA.csv',index_col='Date')
# df=pd.read_csv('data1/DRREDDY.csv',index_col='Date')
df=pd.read_csv('data1/GLENMARK.csv',index_col='Date')
# df=pd.read_csv('data1/ASHOKLEY.csv',index_col='Date')

df=df.drop(columns=[
        'H-L','weekday', 'month', 'day', 'year', 'week', 'o-c', 'c-l', 'h-o', 'h-c','o-l',
                    'updown',
                    'Turnover1',
       'Open_interest1',
    'Change_in_oi1',
 'Turnover2', 'Open_interest2',
       'Change_in_oi2', 'Turnover3', 'Open_interest3', 'Change_in_oi3'
    ])


df['v*TQPT']=df['Volume']*df['trd_qty_per_trade']
df['DV*TQPT']=df['Deliverable Volume']*df['trd_qty_per_trade']
df['fut_to_eq_turnover']=df['Fut_turnover']/df['Turnover']


def data_creation(df):
    df_for_training = df.astype(float)
    df_for_training=df.drop(columns=['updown_after_gap'])
    df_for_training1=df['updown_after_gap']

    print('data taken from csv ')

    scaler = StandardScaler()
    scaler = scaler.fit(df_for_training)
    df_for_training_scaled = scaler.transform(df_for_training)


    print('this is the final df ')
    print(df.columns)
    print(df.tail(100))
    print(df.info())

    trainX = []
    trainY = []

    n_future = 0  # Number of days we want to predict into the future
    n_past = 1 # Number of past days we want to use to predict the future

    for i in range(n_past, len(df_for_training_scaled) - n_future + 1):
        trainX.append(df_for_training_scaled[i - n_past:i, 0:df_for_training.shape[1]])
        trainY.append(df_for_training1[i + n_future - 1:i + n_future,])
        # print('y_train is this ',df_for_training1[i + n_future - 1:i + n_future])
    trainX, trainY = np.array(trainX), np.array(trainY)

    print('trainX shape == {}.'.format(trainX.shape))
    print('trainY shape == {}.'.format(trainY.shape))

    # define Autoencoder model

    print('trainX')
    print(trainX)
    print('trainY')
    print(trainY)

    return trainX,trainY




print('columns of df are this ',df.columns)
i=1

def do():
    df['Open1'] = talib.ROC(df['Open'], timeperiod=i)
    df['Low1'] = talib.ROC(df['Low'], timeperiod=i)
    df['High1'] = talib.ROC(df['High'], timeperiod=i)
    df['Close1'] = talib.ROC(df['Close'], timeperiod=i)
    df['VWAP1'] = talib.ROC(df['VWAP'], timeperiod=i)
    df['Volume1'] = talib.ROC(df['Volume'], timeperiod=i)
    df['Turnover1'] = talib.ROC(df['Turnover'], timeperiod=i)
    df['Trades1'] = talib.ROC(df['Trades'], timeperiod=i)
    df['Deliverable Volume1'] = talib.ROC(df['Deliverable Volume'], timeperiod=i)
    df['%Deliverble1'] = talib.ROC(df['%Deliverble'], timeperiod=i)
    df['Fut_turnover1'] = talib.ROC(df['Fut_turnover'], timeperiod=i)
    df['trd_qty_per_trade1'] = talib.ROC(df['trd_qty_per_trade'], timeperiod=i)
    df['oi1'] = talib.ROC(df['oi'], timeperiod=i)

    # df['Open'] = talib.ROC(df['Open'], timeperiod=i)
    # df['Low'] = talib.ROC(df['Low'], timeperiod=i)
    # df['High'] = talib.ROC(df['High'], timeperiod=i)
    # df['Close'] = talib.ROC(df['Close'], timeperiod=i)
    # df['VWAP'] = talib.ROC(df['VWAP'], timeperiod=i)
    # df['Volume'] = talib.ROC(df['Volume'], timeperiod=i)
    # df['Turnover'] = talib.ROC(df['Turnover'], timeperiod=i)
    # df['Trades'] = talib.ROC(df['Trades'], timeperiod=i)
    # df['Deliverable Volume'] = talib.ROC(df['Deliverable Volume'], timeperiod=i)
    # df['%Deliverble'] = talib.ROC(df['%Deliverble'], timeperiod=i)
    # df['Fut_turnover'] = talib.ROC(df['Fut_turnover'], timeperiod=i)
    # df['trd_qty_per_trade'] = talib.ROC(df['trd_qty_per_trade'], timeperiod=i)
    # df['oi'] = talib.ROC(df['oi'], timeperiod=i)
do()





def create_shifted_data(step):


    df[str(step)+'Close']=df['Close1'].shift(step)






def create_sma_columns(i):
    list=[]
    df['Open'+str(i)] = talib.SMA(df['Open'], timeperiod=i)
    df['Low'+str(i)] = talib.SMA(df['Low'], timeperiod=i)
    df['High'+str(i)] = talib.SMA(df['High'], timeperiod=i)
    df['Close'+str(i)] = talib.SMA(df['Close'], timeperiod=i)
    df['VWAP'+str(i)] = talib.SMA(df['VWAP'], timeperiod=i)
    df['Volume'+str(i)] = talib.SMA(df['Volume'], timeperiod=i)
    df['Turnover'+str(i)] = talib.SMA(df['Turnover'], timeperiod=i)
    df['Trades'+str(i)] = talib.SMA(df['Trades'], timeperiod=i)
    df['Deliverable Volume'+str(i)] = talib.SMA(df['Deliverable Volume'], timeperiod=i)
    df['%Deliverble'+str(i)] = talib.SMA(df['%Deliverble'], timeperiod=i)
    df['Fut_turnover'+str(i)] = talib.SMA(df['Fut_turnover'], timeperiod=i)
    df['trd_qty_per_trade'+str(i)] = talib.SMA(df['trd_qty_per_trade'], timeperiod=i)
    df['oi'+str(i)] = talib.SMA(df['oi'], timeperiod=i)

    list.append('Open'+str(i))
    list.append('High'+str(i))
    list.append('Low'+str(i))
    list.append('Close'+str(i))
    list.append('VWAP'+str(i))
    list.append('Volume'+str(i))
    list.append('Turnover'+str(i))
    list.append('Deliverable Volume'+str(i))
    list.append('%Deliverble'+str(i))
    list.append('Fut_turnover'+str(i))
    list.append('trd_qty_per_trade'+str(i))
    list.append('oi'+str(i))

    return list


def create_class_data(i):


    def compare (name):
        # for j in range(len(df)):
        #
        #     if df[name][j] >df[name+'_sma'+str(i)][j]:
        #         df[name+'_sma'+str(i)][j]= 1
        #     else:
        #         df[name+'_sma'+str(i)][j]=0

        df[name+'_sma'+str(i)]=df[name]/df[name+'_sma'+str(i)]

        pass



    i=i
    # df['Open_sma'+str(i)] = talib.SMA(df['Open'], timeperiod=i)
    # df['Low_sma'+str(i)] = talib.SMA(df['Low'], timeperiod=i)
    # df['High_sma'+str(i)] = talib.SMA(df['High'], timeperiod=i)
    df['Close_sma'+str(i)] = talib.SMA(df['Close'], timeperiod=i)
    df['VWAP_sma'+str(i)] = talib.SMA(df['VWAP'], timeperiod=i)
    df['Volume_sma'+str(i)] = talib.SMA(df['Volume'], timeperiod=i)
    df['Turnover_sma'+str(i)] = talib.SMA(df['Turnover'], timeperiod=i)
    df['Trades_sma'+str(i)] = talib.SMA(df['Trades'], timeperiod=i)
    df['Deliverable Volume_sma'+str(i)] = talib.SMA(df['Deliverable Volume'], timeperiod=i)
    df['%Deliverble_sma'+str(i)] = talib.SMA(df['%Deliverble'], timeperiod=i)
    df['Fut_turnover_sma'+str(i)] = talib.SMA(df['Fut_turnover'], timeperiod=i)
    df['trd_qty_per_trade_sma'+str(i)] = talib.SMA(df['trd_qty_per_trade'], timeperiod=i)
    df['oi_sma'+str(i)] = talib.SMA(df['oi'], timeperiod=i)
    df['fut_to_eq_turnover_sma'+str(i)] = talib.SMA(df['fut_to_eq_turnover'], timeperiod=i)
    df['DV*TQPT_sma'+str(i)] = talib.SMA(df['DV*TQPT'], timeperiod=i)
    df['v*TQPT_sma'+str(i)] = talib.SMA(df['v*TQPT'], timeperiod=i)

    # compare('Open')
    # compare('High')
    # compare('Low')
    # compare('Close')
    compare('VWAP')
    compare('Volume')
    compare('Turnover')
    compare('Trades')
    compare('Deliverable Volume')
    compare('%Deliverble')
    compare('Fut_turnover')
    compare('trd_qty_per_trade')
    compare('fut_to_eq_turnover')
    compare('DV*TQPT')
    compare('v*TQPT')



# create_class_data(3)
# create_class_data(5)
# create_class_data(8)
# create_class_data(13)
# create_class_data(21)





# df.to_csv('data1/nn_data.csv')



gap=1



def create_target(df,gap):

    zeros = 1
    ones = 1


    df['updown_after_gap']=0
    for i in range(gap,len(df)-gap):
        if (df['Close'][i-gap]>df['Close'][i]):
            df['updown_after_gap'][i-gap]=0
            zeros=zeros+1
        else:
            df['updown_after_gap'][i-gap]=1
            ones=ones+1

    print('zeros are')
    print(zeros)
    print('ones are')
    print(ones)

    return df


df=create_target(df=df,gap=gap)







# df=df[['Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble','oi','trd_qty_per_trade','Fut_turnover','updown_after_gap']]
# df=df[['Open1','High','Low1', 'Close1', 'VWAP1', 'Volume1', 'Turnover1', 'Trades1', 'Deliverable Volume1', '%Deliverble1','oi1','trd_qty_per_trade1','Fut_turnover1','updown_after_gap']]
# df=df[['Open1','High','Low1', 'Close1', 'VWAP1', 'Volume1', 'Turnover1', 'Trades1', 'Deliverable Volume1', '%Deliverble1','oi1','trd_qty_per_trade1','Fut_turnover1', 'VWAP', 'Volume', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble','oi','trd_qty_per_trade','Fut_turnover','Open', 'High', 'Low', 'Close','updown_after_gap']]
# df=df[['Open1','High1','Low1','Close1','updown_after_gap']]
# df=df[['VWAP1', 'Volume1','updown_after_gap']]
# df=df[['VWAP1','Deliverable Volume1', '%Deliverble1','updown_after_gap']]
# df=df[['VWAP1','trd_qty_per_trade','updown_after_gap']]
# df=df[['VWAP1','Deliverable Volume1','updown_after_gap']]
# df=df[['VWAP1','updown_after_gap']]
# df=df[[ 'Close1', 'Volume1', 'Turnover1', 'Deliverable Volume1', '%Deliverble1','updown_after_gap']]
# df=df[[ 'Close1','updown_after_gap']]
# df=df[['Close1','updown_after_gap']]
# df=df[['1Close','2Close','3Close','4Close','5Close','6Close','7Close','8Close','9Close','10Close','11Close','12Close','13Close','14Close','updown_after_gap']]
# df=df[['1Close','2Close','updown_after_gap']]

# df=df[['VWAP1','Volume1', 'Turnover1', 'Deliverable Volume1', '%Deliverble1','updown_after_gap']]

# df=df[['Turnover1', 'Deliverable Volume1', '%Deliverble1','updown_after_gap']]

# df=df[['Open1','High','Low1', 'Close1', 'VWAP1', 'Volume1', 'Turnover1',  'Deliverable Volume1','%Deliverble1','updown_after_gap']]
# df=df[['Open1','High','Low1', 'Close1', 'VWAP1', 'Volume1', 'Turnover1', 'Trades1', 'Deliverable Volume1', '%Deliverble1','oi1','trd_qty_per_trade1','Fut_turnover1']+create_sma_columns(20)+create_sma_columns(10)+['updown_after_gap']]
# df=df[create_sma_columns(2)+create_sma_columns(3)+create_sma_columns(5)+create_sma_columns(8)+create_sma_columns(13)+create_sma_columns(21)+['updown_after_gap']]

# df=df.drop(columns=['Close1'])

df[df==np.inf]=np.nan
df.fillna(df.mean(), inplace=True)


df = (df - df.min(axis=0)) / (df.max(axis=0) - df.min(axis=0))








print('This is the df used ')
print(df.columns)
print(df.tail(100))



# df.to_csv('data1/nn_data.csv')


df1=df.iloc[:400,:]


testX, testY = data_creation(df=df1)





df=df.iloc[400:-200,:]




trainX, trainY = data_creation(df=df)



# trainX = (trainX - trainX.min(axis=0)) / (trainX.max(axis=0) - trainX.min(axis=0))
# testX = (testX - trainX.min(axis=0)) / (trainX.max(axis=0) - trainX.min(axis=0))









print('model creatinog is happening now')



model = Sequential()
model.add(LSTM(64, activation='tanh', input_shape=(trainX.shape[1], trainX.shape[2]), return_sequences=True))
model.add(LSTM(64, activation='tanh', return_sequences=True))
# model.add(LSTM(64, activation='tanh', return_sequences=True))
# model.add(Dropout(0.2))
# model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation='tanh', return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(2,activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()



print('this is the data fed into model ')

print(trainX)
print(trainY)

print('model traingin of model  is happening now')

# fit model
history = model.fit(trainX, trainY, epochs=20, batch_size=32, validation_split=0.1, verbose=1)





# # plt.plot(history.history['loss'], label='Training loss')
# # plt.plot(history.history['val_loss'], label='Validation loss')
# # plt.legend()
#
# # Forecasting...
# # Start with the last day in training date and predict future...
# n_future = 90  # Redefining n_future to extend prediction dates beyond original n_future dates...
# forecast_period_dates = pd.date_range(list(train_dates)[-1], periods=n_future, freq='1d').tolist()

print(' this are the shapes')
print(testY.shape)
print(trainY.shape)



forecast = model.predict(trainX)
print('this is the forecast array ',forecast)

print('this is the shape of forecast')
print(forecast.shape)


for i in range(0,len(forecast)):
    if forecast[i][0]>forecast[i][1]:
        forecast[i][0]=0
    else:
        forecast[i][0]=1

right=1
wrong=1
profit=0.01
loss=0.01


for i in range(0,len(forecast)-1):
    if float(forecast[i][0] ) == float(trainY[i]):
        right = right + 1
        profit=profit+abs(trainY[i+1])

    else:
        wrong = wrong +1
        loss=loss+abs(trainY[i + 1])





print('for train right and wrong are this ')
print('right = ', right)
print('wroing = ',wrong)
print('accuracy is ',(right)/(right+wrong)*100)
print('profit = ',profit)
print('loss = ',loss)

print('pnl per trade =',(profit-loss)/(right+wrong))


print('comparision of predictions  for training data ')


#
# for i in range(0,350):
#     print(trainX[i],forecast[i])

















forecast = model.predict(testX)  # forecast
print('this is the forecast array ',forecast)

print('this is the shape of forecast')
print(forecast.shape)

for i in range(0,len(forecast)):
    if forecast[i][0]>forecast[i][1]:
        forecast[i][0]=0
    else:
        forecast[i][0]=1

print('test forecast is this' )
print(forecast)


for i in range(0,100):
    print(forecast[i],testY[i])


right=1
wrong=1
profit=0.01
loss= 0.01



for i in range(0,len(forecast)-1):
    if float(forecast[i][0]) == float(testY[i]):

        right = right + 1
        profit=profit+abs(testY[i+1])

    else:
        wrong = wrong +1
        loss=loss+abs(testY[i+1])


print('for test right and wrong are this ')
print('right = ', right)
print('wroing = ',wrong)
print('accuracy is ',(right)/(right+wrong)*100)
print('profit = ',profit)
print('loss = ',loss)

print('pnl per trade =',(profit-loss)/(right+wrong))




print('comparision of predictions for test data  ')


#
# for i in range(0,350):
#     print(testX[i],forecast[i])






model.save('models/nn9')




# # Perform inverse transformation to rescale back to original range
# # Since we used 5 variables for transform, the inverse expects same dimensions
# # Therefore, let us copy our values 5 times and discard them after inverse transform
# forecast_copies = np.repeat(forecast, df_for_training.shape[1], axis=-1)
# y_pred_future = scaler.inverse_transform(forecast_copies)[:, 0]
#
# # Convert timestamp to date
# forecast_dates = []
# for time_i in forecast_period_dates:
#     forecast_dates.append(time_i.date())
#
# df_forecast = pd.DataFrame({'Date': np.array(forecast_dates), 'Open': y_pred_future})
# df_forecast['Date'] = pd.to_datetime(df_forecast['Date'])
#
# original = df[['Date', 'Open']]
# original['Date'] = pd.to_datetime(original['Date'])
# original = original.loc[original['Date'] >= '2020-5-1']
#
# sns.lineplot(original['Date'], original['Open'])
# sns.lineplot(df_forecast['Date'], df_forecast['Open'])
