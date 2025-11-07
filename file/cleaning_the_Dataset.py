from pyexpat import features
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## loading the dataset
df = pd.read_csv(r"E:\new begining\study for private jobs\technical_skills\python and r\krish_data_science_ML\machine_learning_and_data_science\my data\cleaning_dataset\Ridge Lassso Elastic Regression Practicals\Algerian_forest_fires_dataset_UPDATE.csv",header =1)

df.head()
df.tail()

df.dtypes
df.shape
df.info()
df.describe()
df.columns

## Check for missing values
df.isnull().sum()
df.isnull().values.any()
df.isnull()
df.shape

## removing duplicate rows
duplicate_rows = df.duplicated()
print("Number of duplicate rows:", duplicate_rows.sum())



## showing null values row wise
null_values_row_wise = df[df.isnull().any(axis=1)]
null_values_row_wise.shape
null_values_row_wise

## creating new column based on region
df.loc[:122,"Region"]= 0
df.loc[123:,"Region"]= 1

## changing the data type of Region column to integer
df["Region"] = df["Region"].astype(int)

df.head()

df.loc[122:124,:]

## removing null values and reseting  the index
df = df.dropna().reset_index(drop=True)
df.shape
df.isnull().sum()

## removing 122nd row and reseting the index
df = df.drop(index=122).reset_index(drop=True)

df.info()
df.tail()

##checking column spaces
df.columns

## fixing the column spaces issue
df.columns = df.columns.str.strip()

## changing the required column data types
df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']] = df[['day', 'month', 'year', 'Temperature', 'RH', 'Ws']].astype(int)

df.dtypes

df.head()

## checking columns data types as object
objects = [features for features in df.columns if df[features].dtypes == 'object']
objects

features_1 = [features_1 for features_1 in df.columns if df[features_1].dtypes == "int64"]

features_1

### changing  the remaining object data types to float data types
for i in objects:
    if i != 'Classes':
        df[i] = df[i].astype(float)

df.dtypes

## final cleaned dataset
df.head()

## decribing the cleaned dataset
df.describe()

## saving the cleaned dataset to new csv file
df.to_csv(r"E:\new begining\study for private jobs\technical_skills\python and r\krish_data_science_ML\machine_learning_and_data_science\my data\cleaning_dataset\saving_cleaned_data.csv", index=False)
df.to_csv (r"E:\new begining\study for private jobs\technical_skills\python and r\krish_data_science_ML\machine_learning_and_data_science\my data\cleaning_dataset\clearned_data\save_cleaned_data.csv", index=False)