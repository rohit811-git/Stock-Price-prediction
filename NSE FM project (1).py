#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import the libraries
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


#Get the stock quote
df = web.DataReader('HDFCBANK.NS', data_source='yahoo', start='2000-01-01', end='2020-03-31')


# In[3]:


# Show the data
df


# In[4]:


#Get the number of rows and columns in the data set
df.shape


# In[5]:


#Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price INR (Rs)', fontsize=18)
plt.show()


# In[6]:


#Create a new dataframe with only the 'Close column
data = df.filter(['Close'])


# In[7]:


#Convert the dataframe to a numpy array
dataset = data.values


# In[8]:


#Get the number of rows to train the model on
training_data_len = math.ceil( len(dataset) * .7 )


# In[9]:



training_data_len


# In[10]:


#Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data


# In[11]:


#Create the training data set
#Create the scaled training data set
train_data = scaled_data[0:training_data_len , :]


# In[12]:


#Split the data into x_train and y_train data sets
x_train = []
y_train = []


# In[13]:


for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i<= 61:
    print(x_train)
    print(y_train)
    print()


# In[14]:


#Convert the x_train and y_train to numpy arrays 
x_train, y_train = np.array(x_train), np.array(y_train)


# In[15]:


#Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_train.shape


# In[16]:


#Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences= False))
model.add(Dense(25))
model.add(Dense(1))


# In[17]:


#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[18]:


#Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)


# In[19]:


#Create the testing data set
#Create a new array containing scaled values from index 1543 to 2002 
test_data = scaled_data[training_data_len - 60: , :]


# In[20]:


#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])


# In[21]:


#Convert the data to a numpy array
x_test = np.array(x_test)


# In[22]:


#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))


# In[23]:


#Get the models predicted price values 
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)


# In[24]:


#Get the root mean squared error (RMSE)
rmse=np.sqrt(np.mean(((predictions- y_test)**2)))
rmse


# In[25]:


#Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions


# In[26]:


#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price INR (Rs)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Actual Value', 'Predictions'], loc='lower right')
plt.show()


# In[27]:


#Show the valid and predicted prices
valid


# In[28]:


Reliance_quote = web.DataReader('RELIANCE.NS', data_source='yahoo', start='2019-01-01', end='2020-03-31')


# In[29]:


new_df = Reliance_quote.filter(['Close'])


# In[30]:


last_60_days = new_df[-60:].values


# In[31]:


last_60_days_scaled = scaler.transform(last_60_days)


# In[32]:


X_test = []


# In[33]:


X_test.append(last_60_days_scaled)


# In[34]:


X_test = np.array(X_test)


# In[35]:


X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[36]:


pred_price = model.predict(X_test)


# In[37]:


pred_price = scaler.inverse_transform(pred_price)


# In[38]:


print(pred_price)


# In[39]:


NSE_quote2 = web.DataReader('RELIANCE.NS', data_source='yahoo', start='2020-03-31', end='2020-03-31')


# In[40]:


print(NSE_quote2['Close'])


# In[ ]:




