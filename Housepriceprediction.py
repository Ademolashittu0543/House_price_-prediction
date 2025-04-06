#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[3]:


df = pd.read_csv('data.csv')
df


# In[4]:


df.isna().sum()


# In[5]:


current_year = 2025 
df['house_age'] = current_year - df['yr_built']
df


# In[6]:


numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("numeric variables:")
print(numeric_cols)


# In[7]:


categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

print("Categorical variables:")
print(categorical_cols)


# In[8]:


cat_column_to_use= df["city"]
columns_to_drop=['date','street','statezip','country','yr_renovated','yr_built','city']
df_num=df.drop(columns_to_drop,axis=1)
df_num


# In[9]:


numeric_cols = df_num.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("numeric variables:")
print(numeric_cols)


# In[10]:


Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df_num[~((df_num < lower_bound) | (df_num > upper_bound)).any(axis=1)]

print(f"Original rows: {df.shape[0]}, Cleaned rows: {df_cleaned.shape[0]}")


# In[11]:


df_cleaned


# In[12]:


df_final = df_cleaned.copy()
df_final["city"]=cat_column_to_use

print("\nDataFrame with 'city' restored:")
df_final


# In[13]:


df_final.info()


# In[14]:


corr_matrix = df_cleaned[['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot','sqft_basement', 'floors', 'condition', 'house_age']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[15]:


plt.figure(figsize=(12, 6))
sns.scatterplot(x='sqft_living', y='price', data=df)
plt.title('Price vs Living Area')
plt.show()


# In[16]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='bedrooms', y='price', data=df)
plt.title('Price Distribution by Bedrooms')
plt.show()


# In[17]:


plt.figure(figsize=(12, 6))
sns.barplot(x='floors', y='price', data=df)
plt.title('Price Distribution by floors')
plt.show()


# In[18]:


plt.figure(figsize=(12, 6))
sns.boxplot(x='bathrooms', y='price', data=df)
plt.title('Price Distribution by floors')
plt.show()


# In[19]:


x=df_final.drop(columns=['price'],axis=1)
y=df_final['price']

y=y.round().astype(int)
x


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[21]:


from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline


numeric_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                   'waterfront', 'view', 'condition', 'sqft_above', 
                   'sqft_basement', 'house_age']
categorical_features = ['city'] 


city_categories = [X_train['city'].unique()]


preprocessor = make_column_transformer(
    (OneHotEncoder(categories=city_categories, handle_unknown='ignore'), categorical_features),
    (StandardScaler(), numeric_features),
    remainder='passthrough'
)


pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('Regressor', RandomForestRegressor())
])


pipeline.fit(X_train, y_train)


# In[22]:


pipeline.score(X_test,y_test)


# In[23]:


y_pred=pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.2f}")  
print(f"RMSE: ${(mse**0.5):,.0f}")  


# In[24]:


import pickle

# Assuming 'model' is your trained regression model
with open('modell.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

