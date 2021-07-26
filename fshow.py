from datetime import date
from nsepy import get_history
import pandas as pd
import numpy as np
import talib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics



from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

# import Date
import pickle
import warnings
import sys


import warnings
import sys
if not sys.warnoptions:
    warnings.simplefilter("ignore")



warnings.filterwarnings("ignore")
if not sys.warnoptions:
    warnings.simplefilter("ignore")


from pprint import pprint
from GoogleNews import GoogleNews



pd.set_option("display.max_rows", None, "display.max_columns", None)

stock_names = ['TATAMOTORS', 'HDFC', 'CIPLA','DRREDDY']

stock_name = 'TATAMOTORS'




# df=pd.read_csv('show_result/'+str(stock_name)+'.csv',parse_dates=True)
df=pd.read_csv('show_result/'+str(stock_name)+'ready.csv',parse_dates=True)




day=len(df)-1

print('prediction on date ',df['Date'][day])
print('of '+str(stock_name))

print()




if df['predicted1'][day] ==0 :

    print('preiction by algorithm 2:','DOWN')
    print('DOWN')
else:

    print('preiction by algorithm 2:','UP')






if df['predicted2'][day] ==0 :

    print('preiction by algorithm 2:','DOWN')
    print('DOWN')
else:

    print('preiction by algorithm 2:','UP')


# print(df['predicted2'][len(df)-1])


print()
print()
if df['predictable'][day]==1:
    print('It  is predictable')
else:

    print(' It Is not predictable')





keyword=stock_name



# googlenews=GoogleNews()
#googlenews = GoogleNews(start='03/01/2020',end='03/02/2021',lang='en')



googlenews=GoogleNews('en','100d')


googlenews.get_news(keyword)
# print(googlenews.get_texts())


data=googlenews.results(sort=True)

# print('this is first data ',data[3])

print()
print()

print( str(stock_name))

print('Recent News related to ',str(stock_name)+' is = '+data[0]['title'])
print()

print('on date')
print('  '+str(data[0]['datetime']))



# print('each elements')
# print('')
# print('title is  ',data[0]['title'])
# print('date is ',data[0]['date'])
# print('discription is ',data[0]['desc'])
# pprint(data[2]['desc'])




import pickle



from keras.models import load_model
from keras.models import save_model
# model= model.load('all_models/'+str(stock_name)+'.h5')


# model_loaded=load_model('proj3/all_models/'+str(stock_name)+'.h5')
model_loaded=load_model('E:/stock_project/proj3/all_models/'+str(stock_name)+'.h5')






# model_loaded.predict(df[''])











#
# keyword=stock_name
#
#
#
# # googlenews=GoogleNews()
# #googlenews = GoogleNews(start='03/01/2020',end='03/02/2021',lang='en')
#
#
#
# googlenews=GoogleNews('en','100d')
#
#
# googlenews.get_news(keyword)
# # print(googlenews.get_texts())
#
#
# data=googlenews.results(sort=True)
#
# # print('this is first data ',data[3])
#
#



# model_loaded=load_model('E:/stock_project/proj3/all_models/'+str(stock_name)+'.h5')

df=pd.read_csv('E:/stock_project/proj3/data1/'+str(stock_name)+'.csv')



df['Headlines'].replace("[^a-zA-Z]", " ",regex=True,inplace=True)


tokenizer = Tokenizer(num_words=5000, split=" ")
# tokenizer.fit_on_texts(df['Headlines'].values)
tokenizer.fit_on_texts(data[0]['title'])


# tokenizer.fit_on_text(data[0]['title'])

# model_loaded.predict(data[0]['title'])
X = tokenizer.texts_to_sequences(data[0]['title'])
X = pad_sequences(X)

sentiment = model_loaded.predict(X)



if sentiment[0][0] >0.5:
    print('sentiment  of news is = ', 'positive')
else:
    print('sentiment  of news is ', 'negative')














