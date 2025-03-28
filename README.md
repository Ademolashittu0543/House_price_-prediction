# House Price Prediction  

## Objective  
To clean the dataset by identifying and removing outliers using the Interquartile Range (IQR) method while ensuring all relevant columns are numeric

## Dataset  
The dataset contains multiple numerical features, but some columns might have been stored as non-numeric (string) values, leading to errors when performing numerical operations

## Methods  
- Checked Data Types – Used df.dtypes to identify non-numeric columns in the dataset
- Converted to Numeric – Applied pd.to_numeric(errors='coerce') to convert non-numeric values to numerical, replacing invalid entries with NaN
- Computed IQR – Calculated the 1st quartile (Q1) and 3rd quartile (Q3) and derived the IQR (Q3 -Q1) for each numerical column
- Defined Outlier Bounds – Set the lower and upper bounds as Q1 - 1.5 × IQR and Q3 + 1.5 × IQR, respectively
- Filtered Out Outliers – Removed rows containing values outside these bounds in any column

## Challenges  
- Handling missing values
- Encountered an error due to non-numeric values in some columns
- Had to convert all necessary columns to numeric before applying the IQR method 
- Feature selection for better model accuracy  

## What I Learned  
- How to clean and preprocess data using Pandas  
- How to train a regression model in Python  
- The importance of checking data types before performing numerical operations
- How to handle mixed-type datasets using pd.to_numeric(errors='coerce')
- The IQR method is an effective technique for removing extreme outliers  

## Next Steps  
- Improve model accuracy with feature scaling  
  
