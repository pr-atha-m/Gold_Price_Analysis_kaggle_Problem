#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import seaborn as sns  # For data visualization 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np  # For mathematical calculations 
import warnings   # To ignore any warnings 
warnings.filterwarnings("ignore")


# In[2]:


#importing Data set Of GOld_prices (.csv_format)
data = pd.read_csv('gld_price_data.csv')
data


# In[3]:


#descriptives of dataset
data.describe()


# In[4]:


#information Of DataSet
data.info()


# In[5]:


#No. of rows and columns in our data_set
data.shape


# In[6]:


#checking if there is a missing Value 
data.isnull().sum()


# In[7]:


#heat map to understand correlation between variables
corr = data.corr()
plt.figure(figsize = (7,6))
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            annot=True,fmt='.2f',linewidths=0.30 ,)
plt.title('Correlation of Features in Data', y = 1.05, size=15)


# Gold_prices are highly correlated with SLV

# In[8]:


#Now we check the relation of GLD variable with SLV
sns.jointplot(x =data['SLV'], y = data['GLD'], color = 'purple')


# Our target variable is GLD ie. the gold_prices

# In[9]:


#checking correlation values of GLD
print(corr['GLD'])


# In[10]:


#visualising distribution Of Gold Prices
sns.distplot(data['GLD'] , color = 'red')


# GLD price is quite uniformly Distributed

# In[11]:


# Extracting years from date
time = pd.DatetimeIndex(data['Date'])
plt.figure(figsize = (13,10) , dpi = 100)

data['years'] = time.year
plt.subplot(2 , 2 , 1)
data.groupby('years')['GLD'].max().plot(kind = 'bar' , color = 'red')
plt.subplot(2 , 2 , 2)
data.groupby('years')['GLD'].min().plot(kind = 'bar' , color = 'green')
plt.subplot(2 , 2 , 3)
data.groupby('years')['GLD'].mean().plot(kind = 'bar' , color = 'yellow')


# Here we have plotted Bar Graphs for max , min and mean of gold_prices from 2008 to 2018.

# In[12]:


#plotting Gold Price Trend over different Years
plt.plot(data['Date'] , data['GLD'] , color = 'green')
plt.xlabel('Years (2008 - 2018) ')
plt.ylabel('Gold Prices (USD) ')
plt.show()


# In[13]:


data.drop(columns = ['Date' , 'years'] , inplace = True , axis = 1)


# In[14]:


data.head(5)


# # Training Our Model Using Random Forest

# In[15]:


X = data.drop(columns = ['GLD'] , axis = 1)
Y = data['GLD']


# In[16]:


print(X)


# In[17]:


print(Y)


# In[18]:


#Spliting the dataset into training and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# In[19]:


#Now fitting the Random forest regression to the traning set
#Import Random Forest Model
from sklearn.ensemble import RandomForestRegressor
#Create a Gaussian Classifier
regressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
#Train the model using the training sets y_pred=clf.predict(X_test)
regressor.fit(x_train, y_train)


# In[20]:


y_pred = regressor.predict(x_test)


# In[21]:


y_pred.shape


# In[22]:


y_test.shape


# In[23]:


#score of our Model
regressor.score(x_test , y_test)


# In[25]:


y_test = list(y_test)


# In[27]:


#Visualising Predictions vs Actual
plt.plot(y_test, color = 'green', label = 'Acutal')
plt.plot(y_pred, color = 'deeppink', label = 'Predicted')
plt.grid(0.3)
plt.title('Acutal vs Predicted')
plt.xlabel('Number of Oberservation')
plt.ylabel('GLD')
plt.legend()
plt.show()


# In[ ]:




