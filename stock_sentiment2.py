import pandas as pd
from datetime import datetime
# from pprint import pprint
import numpy as np
import sklearn

import joblib
from keras.models import load_model
from keras.models import save_model
pd.set_option("display.max_rows", None, "display.max_columns", None)


#
# stock_name='TATAMOTORS'
# stock_name='BPCL'
# stock_name='GAIL'
# stock_name='CIPLA'
# stock_name='AXISBANK'
stock_name='HDFC'


# df=pd.read_pickle('data1/TATAMOTORS.pkl')
# df=pd.read_pickle('data1/'+str(stock_name)+'.pkl')
df=pd.read_csv('data1/'+str(stock_name)+'.csv')
print(df)

df['rough']=0
for i in range(0,len(df)-1):
    if df['Date1'][i]==df['Date1'][i+1]:
        df['rough'][i+1]=1




#created updown after gap

df['updown_after_gap']=0
for i in range(0,len(df)-1):
    if df['percentage_change'][i]>0:

        df['updown_after_gap'][i]=1


zeros=1
ones=1
for i in range(0,len(df)-1):
    if df['updown_after_gap'][i]==0:
        zeros=zeros+1
    else:
        ones=ones+1

print('zeros are',zeros)
print('Ones  are',ones)



print(df)




import re
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Embedding, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.60
# session = tf.Session(config=config)
#
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.60)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))










df['Headlines'].replace("[^a-zA-Z]", " ",regex=True,inplace=True)




tokenizer = Tokenizer(num_words=5000, split=" ")
tokenizer.fit_on_texts(df['Headlines'].values)

X = tokenizer.texts_to_sequences(df['Headlines'].values)
X = pad_sequences(X) # padding our text vector so they all have the same length



model = Sequential()
model.add(Embedding(5000, 256, input_length=X.shape[1]))

model.add(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(32, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


y= pd.get_dummies(df['updown_after_gap']).values




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

batch_size = 256
epochs = 20



print('length of test data is ',len(X_test))





model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)


import pickle
# from sklearn import joblib
# import joblib
# from sklearn.externals import joblib

# saved_model=pickle.dump(model)

#
# pkl_filename = "tata.pkl"
# with open(pkl_filename, 'wb') as file:
#     pickle.dump(model, file)

model.save('all_models/'+str(stock_name)+'.h5')






print('accurcy on test dataset is ')
print(' ')


score, acc = model.evaluate(X_test,y_test,
                            batch_size=batch_size)


print(score,acc)




model_loaded=load_model('all_models/'+str(stock_name)+'.h5')

score, acc = model_loaded.evaluate(X_test,y_test,
                            batch_size=batch_size)


print(score,acc)













