#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[2]:


df = pd.read_csv('data.csv')
df


# In[3]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])


# In[4]:


print(df.dtypes)

df = df.apply(pd.to_numeric, errors='coerce')

Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]

print(f"Original rows: {df.shape[0]}, Cleaned rows: {df_cleaned.shape[0]}")
df_cleaned


# In[5]:


current_year = 2025 
df_cleaned['house_age'] = current_year - df_cleaned['yr_built']

columns_to_drop=['date','street','statezip','country','yr_renovated','sqft_above','yr_built','city']
df_cleaned=df_cleaned.drop(columns_to_drop,axis=1)
df_cleaned


# In[6]:


corr_matrix = df_cleaned[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','sqft_basement', 'floors', 'condition', 'house_age']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[7]:


plt.figure(figsize=(12, 6))
sns.scatterplot(x='sqft_living', y='price', data=df)
plt.title('Price vs Living Area')
plt.show()


# In[8]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='bedrooms', y='price', data=df)
plt.title('Price Distribution by Bedrooms')
plt.show()


# In[9]:


plt.figure(figsize=(12, 6))
sns.barplot(x='floors', y='price', data=df)
plt.title('Price Distribution by floors')
plt.show()


# In[10]:


x=df_cleaned.drop(columns=['price'],axis=1)
y=df_cleaned['price']

y=y.round().astype(int)
x


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[12]:


steps = [
    ('scaler', StandardScaler()),
    ('regressor', RandomForestRegressor())
]
rf_pipeline = Pipeline(steps)
rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)


# In[13]:


mse = mean_squared_error(y_test, y_pred_rf)
r2 = r2_score(y_test, y_pred_rf)

print(f"R² Score: {r2:.2f}")  
print(f"RMSE: ${(mse**0.5):,.0f}")  


# In[14]:


from xgboost import XGBRegressor
steps = [
    ('scaler', StandardScaler()),
    ('regressor', XGBRegressor())
]
XGB_pipeline = Pipeline(steps)
XGB_pipeline.fit(X_train, y_train)
y_pred_XGB = rf_pipeline.predict(X_test)


# In[15]:


mse = mean_squared_error(y_test, y_pred_XGB)
r2 = r2_score(y_test, y_pred_XGB)

print(f"R² Score: {r2:.2f}")  
print(f"RMSE: ${(mse**0.5):,.0f}")  

