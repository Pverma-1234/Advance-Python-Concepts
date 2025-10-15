#Steps Of Preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/User/Downloads/Data.csv")
print("Original data:\n", df)

#Handling Missing data
#Filling missing age with mean
df["Age"].fillna(df["Age"].mean(),inplace=True)

#Filling Missing Salary with median
df["Salary"].fillna(df["Salary"].median(),inplace=True)

print("Data after Handling missing value:\n ", df)

#Encoding Categorical Data
#Encode Gender  (Ordinal, Male=1,Female=0)
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

#Encode Purchased (Yes=1,NO = 0)
le_purchase = LabelEncoder()
df['Purchased'] = le_purchase.fit_transform(df['Purchased'])

#One-Hot Encode Country (Non-Ordinal)
df=pd.get_dummies(df,columns=['Country'])
print("Data after encoding categorical variables:\n",df)


#Feature Scaling mean = 0 ,std deviation =1, -> to remove baiseness
#Apply on numeric data no range for scaling

scaler = StandardScaler()
df[['Age','Salary']] = scaler.fit_transform(df[['Age','Salary']])
print("Data After Scaling:\n",df)


#Handling Outliers
#using IQR method on salary
#Find Q1,Q3 and IQR

Q1=df['Salary'].quantile(0.25)
Q3=df['Salary'].quantile(0.75)
IQR = Q3-Q1
 
#Define lower and upper limits
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

#Find Outliers
outliers = df[(df['Salary']<lower_limit) | (df['Salary'] > upper_limit)]

print("Outliers:\n",outliers)

#Remove Outlier
clean_data = df[(df['Salary']>= lower_limit) & (df['Salary']<=upper_limit)]

print("Data after removing outliers:\n",clean_data)
