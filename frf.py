import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import talib
# import Date
import pickle

import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")

pd.set_option("display.max_rows", None, "display.max_columns", None)
gap=5


# df=pd.read_csv('data1/TATAMOTORS.csv',index_col='Date')

# df=pd.read_csv('data1/EICHERMOT.csv',index_col='Date')
# df=pd.read_csv('data1/BAJAJ-AUTO.csv',index_col='Date')
# df=pd.read_csv('data1/stock_eq_fut.csv',index_col='Date')
# df=pd.read_csv('data1/GAIL.csv',index_col='Date')
# df=pd.read_csv('data1/CIPLA.csv',index_col='Date')
# df=pd.read_csv('data1/AXISBANK.csv',index_col='Date')
# df=pd.read_csv('data1/HDFC.csv',index_col='Date')
df=pd.read_csv('data1/COALINDIA.csv',index_col='Date')

df=df.iloc[:-200,:]

df.reset_index(inplace=True)


df['Date']=df['Date'].apply(pd.to_datetime)

df.reset_index(inplace=True)

df['month']=df.Date.dt.month
df['day']=df.Date.dt.day
df['year']=df.Date.dt.year
df['week']=df.Date.dt.week



df['change_oi'] = talib.ROC(df['oi'], timeperiod=1)
df['chg_percentage'] = talib.ROC(df['Close'], timeperiod=gap)
df['1d_chg_percentage'] = abs(talib.ROC(df['Close'], timeperiod=1))
# df['fut_turnover_chg']=talib.ROC(df['Fut_turnover'],timeperiod=1)

df['atr'] = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=14)
df['stdev'] = talib.STDDEV(df['Close'], timeperiod=14, nbdev=1)

df['stdev_avg']= talib.SMA(df['stdev'],timeperiod=60)
df['atr_avg']= talib.SMA(df['atr'],timeperiod=60)



df['min21'],df ['max21'] = talib.MINMAX(df['Close'], timeperiod=21)
df['min51'],df ['max51'] = talib.MINMAX(df['Close'], timeperiod=52)

df['min21']=df['min21']/df['Close']
df['max21']=df['max21']/df['Close']
df['min51']=df['min51']/df['Close']
df['max51']=df['max51']/df['Close']


for i in range(2,121,7):
    df['closed_sma'+str(i)]=talib.SMA(df['Close'],timeperiod=i)/df['Close']
    # df['open_sma'+str(i)]=talib.SMA(df['Open'],timeperiod=i)/df['Open']
    # df['low_sma'+str(i)]=talib.SMA(df['Low'],timeperiod=i)/df['Low']
    # df['High_sma'+str(i)]=talib.SMA(df['High'],timeperiod=i)/df['High']



# df['fut_to_equity_turnover']=df['Fut_turnover']/df['Turnover']
df['v*TQPT']=df['Volume']*df['trd_qty_per_trade']
df['DV*TQPT']=df['Deliverable Volume']*df['trd_qty_per_trade']
















################################################################## new Features changes $##########################################################



# df['fut_to_equity_turnover']=df['Fut_turnover']/df['Turnover']
df['v*TQPT']=df['Volume']*df['trd_qty_per_trade']
df['DV*TQPT']=df['Deliverable Volume']*df['trd_qty_per_trade']




# for i in range(2,11,3):
#
#     df['sma_'+str(i)+'PC']= talib.SMA(df['percentage_change'],timeperiod=i)
    # df['fut_to_equity_turnover' + str(i)] = talib.SMA(df['fut_to_equity_turnover'], timeperiod=i)/df['fut_to_equity_turnover']
    # df['v*TQPT'+str(i)] = (talib.SMA(df['v*TQPT'], timeperiod=i))/df['v*TQPT']
    # df['DV*TQPT'+str(i)] = (talib.SMA(df['DV*TQPT'], timeperiod=i))/df['DV*TQPT']
    #
    # pass
#
#
#
# for i in range(2,51,7):
#
#     # df['sma_'+str(i)+'PC']= talib.SMA(df['percentage_change'],timeperiod=i)/(df['percentage_change']+0.001)
#     df['sma_'+str(i)+'DV']= talib.SMA(df['Deliverable Volume'],timeperiod=i)/df['Deliverable Volume']
#     df['sma_'+str(i)+'TQPT']= talib.SMA(df['trd_qty_per_trade'],timeperiod=i)/df['trd_qty_per_trade']
#     df['%Deliverble' + str(i)] = talib.SMA(df['%Deliverble'], timeperiod=i)/df['%Deliverble']
#     df['Fut_turnover' + str(i)] = talib.SMA(df['Fut_turnover'], timeperiod=i)/df['Fut_turnover']
#     # df['fut_to_equity_turnover' + str(i)] = talib.SMA(df['fut_to_equity_turnover'], timeperiod=i)/df['fut_to_equity_turnover']
#     df['Volume'+str(i)] = talib.SMA(df['Volume'], timeperiod=i)/df['Volume']
#
#     df['oi'+str(i)] = talib.SMA(df['oi'], timeperiod=i)/df['oi']
#
#
#     pass
#
#
# pass
#
#
# for i in range(2,50,3):
#
#     df['fut_to_equity_turnover' + str(i)] = talib.SMA(df['fut_to_equity_turnover'], timeperiod=i)
#     # df['PC' + str(i)] = talib.ROC(df['percentage_change'], timeperiod=i)
#     pass
#
#
#
#
#
# for i in range(2,51,5):
#     df['Volume'+str(i)] = talib.SMA(df['Volume'], timeperiod=i)
#     # df['Volume'+str(i)] = talib.ROC(df['Volume'], timeperiod=i)
#     # df['fut_turnover_chg'+str(i)] = talib.SMA(df['Fut_turnover'], timeperiod=i)
#     # df['oi' + str(i)] = talib.SMA(df['oi'], timeperiod=i)
#     # df['%Deliverble' + str(i)] = talib.SMA(df['%Deliverble'], timeperiod=i)
#     # df['fut_to_equity_turnover' + str(i)] = talib.ROC(df['fut_to_equity_turnover'], timeperiod=i)
#     # df['trd_qty_per_trade' + str(i)] = talib.ROC(df['trd_qty_per_trade'], timeperiod=i)
#     # df['Deliverable Volume' + str(i)] = talib.ROC(df['Deliverable Volume'], timeperiod=i)
#     df['Close' + str(i)] = talib.SMA(df['Close'], timeperiod=i)
#     df['Open' + str(i)] = talib.SMA(df['Open'], timeperiod=i)
#     pass
#
#
#










df['updown_after_gap']=0
for i in range(gap,len(df)-gap):
    if (df['Close'][i-gap]>df['Close'][i]):
        df['updown_after_gap'][i-gap]=0
    else:
        df['updown_after_gap'][i-gap]=1


df.set_index('Date',inplace=True)






df[df==np.inf]=np.nan
df.fillna(df.mean(), inplace=True)

df=df.iloc[200:,:]
df=df.iloc[:-50,:]






train=df
train=train.drop(['updown','updown_after_gap' ],axis=1)




# df.to_csv('data1/result_of_prediction.csv')


test = df[['updown_after_gap']]
df = df.dropna()









x_train, x_test, y_train, y_test = train_test_split(train, test, random_state=42,shuffle=True,test_size=0.20)

rfc = RandomForestClassifier(n_estimators=400, oob_score=True, criterion='gini')
rfc.fit(x_train, y_train)





y_pred = rfc.predict(x_test)

rough = pd.DataFrame()
old_accurcay=(metrics.accuracy_score(y_test, y_pred))*100,
print("Accuracy :",old_accurcay, '%')







y_test['predicted']=y_pred
y_test= y_test.rename(
    columns={'updown_after_gap': 'updown_after_gap1'})



df=df.join(y_test)



df.dropna(inplace=True)
df['result']=2
for i in range(len(df)):
    if df['updown_after_gap'][i]==df['predicted'][i]:
        df['result'][i]=1
    else:
        df['result'][i]=0




print(df.head())
print(df['stdev'].mean())
print(df['atr'].mean())
mean_sd=df['stdev'].mean()
mean_atr=df['atr'].mean()









right=1
wrong=1
zero=0
one=0
profit=0
loss=0

for i in range(len(df)-gap):
    if df['updown_after_gap'][i]==0:
        zero+=1
    else:
        one+=1



    if (df['stdev'][i]>0*df['stdev_avg'][i] and df['stdev'][i]<=1*df['stdev_avg'][i]) and (df['atr'][i]>df['atr_avg'][i]*0 and df['atr'][i]<1*df['atr_avg'][i]):



        # if ( ((df['AVG_1d_pc'][i]>0 and  df['AVG_1d_pc'][i]<=1.5) or  (df['AVG_1d_pc'][i]<=0 and  df['AVG_1d_pc'][i]>=-1.5))):
        # if ( ((df['AVG_1d_pc'][i]>1 and  df['AVG_1d_pc'][i]<=1.5) or  (df['AVG_1d_pc'][i]<=1 and  df['AVG_1d_pc'][i]>=-1.5))):
        # if df['stdev'][i]>mean_sd*0 and df['stdev'][i]<=2*mean_sd and df['atr'][i]>mean_atr*0 and df['atr'][i]<2*mean_atr and ( ((df['AVG_1d_pc'][i]>=0 and  df['AVG_1d_pc'][i]<=2.00) or  (df['AVG_1d_pc'][i]<=0 and  df['AVG_1d_pc'][i]>=-2.0))):
        # if df['stdev'][i]>mean_sd*0 and df['stdev'][i]<=2*mean_sd and df['atr'][i]>mean_atr*0 and df['atr'][i]<2*mean_atr and ( ((df['AVG_1d_pc'][i]>=0 and  df['AVG_1d_pc'][i]<=2.00) or  (df['AVG_1d_pc'][i]<=0 and  df['AVG_1d_pc'][i]>=-2.0))):
        # if (df['stdev'][i]>mean_sd*1 and df['stdev'][i]<=2*mean_sd) and (df['atr'][i]>mean_atr*1 and df['atr'][i]<2*mean_atr):
        # if (df['stdev'][i]>mean_sd*0.5 and df['stdev'][i]<=1.5*mean_sd) and (df['atr'][i]>mean_atr*0.5 and df['atr'][i]<1.5*mean_atr):
        # if ( ((df['AVG_1d_pc'][i]>0 and  df['AVG_1d_pc'][i]<=1.5) or  (df['AVG_1d_pc'][i]<=0 and  df['AVG_1d_pc'][i]>=-1.5))):


        if df['result'][i]==1:
            right+=1
            profit+=abs(df['chg_percentage'][i+gap])


        if df['result'][i]==0:
            wrong+=1
            loss+=abs(df['chg_percentage'][i+gap])


print('number of right predictions are: ',right)
print('number of wrong predictions are: ',wrong)
print()
new_accuracy=right/(right+wrong)*100
print('old accuracy is ',old_accurcay)
print('new accuracy is ',new_accuracy)
print()




print('profit is  is ',profit)
print('loss is',loss)
print()


# print('profit per trade is ',(profit-loss)/(right+wrong))