#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import numpy as np
data=pd.read_csv('data.csv')
df=pd.DataFrame(data)
df


# In[2]:


df.isna().sum()


# In[3]:


columns_to_drop=['date','street','city','statezip','country','yr_renovated','price']
x=df.drop(columns_to_drop,axis=1)
x.shape
x


# In[4]:


y = data['price']
y


# In[5]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    x, y, 
                                    test_size=0.2,
                                    random_state=42
                                    )


# In[6]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[7]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)


# In[8]:


y_pred = model.predict(X_test)


# In[9]:


y_pred


# In[ ]:




