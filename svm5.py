###################################################### latest svm #################

#PC=percentage change
#DV=Dilevrable volume
#TQPT=Traded quantity per trade


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import talib
# import Date
import pickle
from sklearn import svm

pd.set_option("display.max_rows", None, "display.max_columns", None)



gap=1


# df=pd.read_csv('data1/stock_eq_fut.csv',index_col='Date')
# df=pd.read_csv('data1/BAJAJ-AUTO.csv',index_col='Date')
df=pd.read_csv('data1/HDFC.csv',index_col='Date')
# df=pd.read_csv('data1/CIPLA.csv',index_col='Date')
# df=pd.read_csv('data1/BPCL.csv',index_col='Date')

df.reset_index(inplace=True)
df=df.drop(['updown' ],axis=1)




df['Date']=df['Date'].apply(pd.to_datetime)
df.reset_index(inplace=True)
df.set_index('Date',inplace=True)






# df=df[['Open','Close','oi','%Deliverble','Fut_turnover','Turnover','trd_qty_per_trade','Deliverable Volume','Volume']]
df=df[['Close','Volume','oi','Fut_turnover','Turnover','Deliverable Volume','percentage_change','trd_qty_per_trade','%Deliverble']]
# df=df[['Close','Volume','Fut_turnover','Turnover','Deliverable Volume']]

df['fut_to_equity_turnover']=df['Fut_turnover']/df['Turnover']
df['v*TQPT']=df['Volume']*df['trd_qty_per_trade']
df['DV*TQPT']=df['Deliverable Volume']*df['trd_qty_per_trade']
df['1d_chg_percentage'] = abs(talib.ROC(df['Close'], timeperiod=1))



for i in range(2,51,7):

    df['Volume'+str(i)] = talib.SMA(df['Volume'], timeperiod=i)/df['Volume']
    df['fut_to_equity_turnover'+str(i)] = talib.SMA(df['fut_to_equity_turnover'], timeperiod=i)/df['fut_to_equity_turnover']
    pass



i=50
df['DV'+str(i)]=(df['Deliverable Volume'])/talib.SMA(df['Deliverable Volume'],timeperiod=i)
df['V'+str(i)]=(df['Volume'])/talib.SMA(df['Volume'],timeperiod=i)
df['%D'+str(i)]=(df['%Deliverble'])/talib.SMA(df['%Deliverble'],timeperiod=i)
df['AVG_change'+str(i)]=(df['percentage_change'])/talib.SMA(df['percentage_change'],timeperiod=i)
df['AVG_oi'+str(i)]=(df['oi'])/talib.SMA(df['oi'],timeperiod=i)
df['AVG_fut_to_eq_tnr'+str(i)]=(df['fut_to_equity_turnover'])/talib.SMA(df['fut_to_equity_turnover'],timeperiod=i)
df['AVG_v*TQPT'+str(i)]=(df['v*TQPT'])/talib.SMA(df['v*TQPT'],timeperiod=i)
df['AVG_DV*TQPT'+str(i)]=(df['DV*TQPT'])/talib.SMA(df['DV*TQPT'],timeperiod=i)










df['updown_after_gap']=2
for i in range(gap,len(df)-gap):
    if (df['Close'][i-gap]>df['Close'][i]):
        df['updown_after_gap'][i-gap]=0
    else:
        df['updown_after_gap'][i-gap]=1







df[df==np.inf]=np.nan
df.fillna(df.mean(), inplace=True)

df=df.iloc[200:,:]
df=df.iloc[:-50,:]








print('columns of df are ',df.columns)

train=df
train=train.drop(['updown_after_gap' ],axis=1)






# train=train.drop(['percentage_change','trd_qty_per_trade','%Deliverble'],axis=1)




print(df.info())



####################################################### applying SVM####################################################

from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics


print(df.head(10))

print(df.columns)

# df=df[['Prev_close', 'Open', 'High', 'Low', 'Close', 'VWAP', 'Volume',
#        'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble', 'Turnover1',
#        'Open_interest1', 'Change_in_oi1', 'Turnover2', 'Open_interest2',
#        'Change_in_oi2', 'Turnover3', 'Open_interest3', 'Change_in_oi3',
#        'Fut_turnover', 'oi', 'percentage_change', 'H-L', 'trd_qty_per_trade',
#        'weekday', 'month', 'day', 'year', 'week', 'o-c', 'c-l', 'h-o', 'h-c',
#        'o-l',
#        'change_oi', 'chg_percentage', 'fut_turnover_chg',
#        'sma2', 'sma5', 'sma8', 'sma11', 'sma14', 'sma17', 'sma20', 'sma23',
#        'sma26', 'sma29', 'sma32', 'sma35', 'sma38', 'sma41', 'sma44', 'sma47',
#        'atr', 'stdev', 'updown_after_gap']]

X=df
X=X.drop(columns=['updown_after_gap'])

y=df[['updown_after_gap']]


print('x ',X.columns)
print('and y are this ',y.columns)




x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=4)

norm=MinMaxScaler().fit(x_train)

x_train=norm.transform(x_train)
x_test=norm.transform(x_test)






print('done')


svm=SVC(kernel='poly',C=300.0,gamma=0.3)

svm.fit(x_train,y_train)


print('accuracy is this ')
svm_confidence=svm.score(x_test,y_test)




y_pred = svm.predict(x_test)

old_accurcay=(metrics.accuracy_score(y_test, y_pred))*100,

print('old accuracy on test data is ',old_accurcay)




###############train accuracy###################################
y_pred = svm.predict(x_train)

old_accurcay=(metrics.accuracy_score(y_train, y_pred))*100,

print('old accuracy of train data is ',old_accurcay)
















#####################################calculating data biased or not########################
ones=0
zeros=0
for i in range(0,len(y_test)):
    if y_test['updown_after_gap'][i]==0:
        zeros=zeros+1
    if y_test['updown_after_gap'][i]==1:
        ones=ones+1

print()
print()
print()
print(zeros,ones)