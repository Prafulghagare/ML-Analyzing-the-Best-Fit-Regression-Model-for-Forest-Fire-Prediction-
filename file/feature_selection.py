## importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

## loading cleaned dataset
df = pd.read_csv(r"E:\new begining\study for private jobs\technical_skills\python and r\krish_data_science_ML\machine_learning_and_data_science\my data\cleaning_dataset\clearned_data\save_cleaned_data.csv")

df.head()
df.tail()

## dropping columns month,day and year in place true
df.columns
df.drop(['day', 'month', 'year'],axis=1,inplace=True)

df.head()

## endoing Classes not fire= 0 and fire =1

df["Classes"].value_counts()

df['Classes'] = np.where(df['Classes'].str.contains("not fire"),0,1)
df['Classes'].value_counts()

## independent and dependent features
## drop FWI from axis =1
x= df.drop("FWI", axis=1)
y= df["FWI"]
y.head()
x.head()

##train and test dataset
from sklearn .model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size= 0.25, random_state=42)

x_train.count()
x_train.shape, x_test.shape

y_train.shape,y_test.shape 
df.shape

## feature selection based on correlation
x_train.corr()

## check of multicollinearity through head map

plt.figure(figsize=(12,10))
corr = x_train.corr()
sns.heatmap(corr,annot=True)
plt.show()

## removing correlation threshold
def correlation (dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])> threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr 


## threshold domain expertise
corr_features = correlation(x_train,0.85)

## dropping features when correlation is more than 0.85
x_train.drop(corr_features,axis=1,inplace= True)
x_test.drop(corr_features,axis = 1, inplace = True)
x_test.shape,x_train.shape

## features scaling or standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

x_test_scaled

## box plot to understand the effects of standared scale

plt.figure(figsize= (15,5))
plt.subplot(1,2,1)
sns.boxplot(data= x_train)
plt.title("x_train_before_scaling")
plt.subplot(1,2,2)
sns.boxplot(data=x_train_scaled)
plt.title("X_train_after_scaling")
plt.show()

