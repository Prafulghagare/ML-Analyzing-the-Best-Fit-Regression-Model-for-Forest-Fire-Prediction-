import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## loading the dataset
df= pd.read_csv(r"E:\new begining\study for private jobs\technical_skills\python and r\krish_data_science_ML\machine_learning_and_data_science\my data\cleaning_dataset\clearned_data\save_cleaned_data.csv" )

df.head()
df.tail()
df.dtypes
df.shape 
df.info()
df.columns

df['Classes'].value_counts()

## dropping unnecessary columns
df_copy = df.drop(['day', 'month', 'year'],axis=1)

df_copy.head()
df_copy.shape

### Check for missing values
df_copy.isnull().sum()

df_copy.columns

## encoding the categorical columns in classes column not_fire=0 ,fire=1
df_copy['Classes'] = np.where(df_copy['Classes'].str.contains("not fire"), 0, 1)

df_copy.head()
df_copy['Classes'].value_counts()
df_copy.tail()

## plot density plot fir all features
sns.set_theme(style="darkgrid") 
df_copy.hist(bins=50, figsize=(16, 12))
plt.show()

## percentage of pie chart
percentage = df_copy["Classes"].value_counts(normalize=True)*100

## ploting pie chart
classlable = ["Fire","Not_Fire"]
plt.figure(figsize=(12,7))
pie_chart = plt.pie(percentage,labels=classlable,autopct="%1.1f%%")

plt.show()
## applying correlation

df_copy.corr()

## applying heat map
sns.heatmap(df_copy.corr())


## monthly fire analysis 

df.head()

df_copy.head()

## linear regression model















