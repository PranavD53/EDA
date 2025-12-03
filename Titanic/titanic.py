import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=sns.load_dataset("titanic")
print(df.head())

print("Shape: ",df.shape)
print(df.info())
print(df.describe(include="all"))

missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_percent": (df.isnull().sum()/len(df))*100
})
missing.sort_values(by="missing_percent", ascending=False)

df['age']=df['age'].fillna(df['age'].mean())
df['embarked']=df['embarked'].fillna(df['embarked'].mode()[0])
df.drop('deck',axis=1,inplace=True)

print("Missing values after cleaning: ",df.isnull().sum().sum())

missing2 = pd.DataFrame({
    "missing_count": df.isnull().sum(), 
    "missing_percent": (df.isnull().sum()/len(df))*100
})
missing2.sort_values(by="missing_percent", ascending=False)

print("Duplicate rows: ",df.duplicated().sum())

df['class']=df['class'].astype('category')
df['embarked']=df['embarked'].astype('category')
df['sex']=df['sex'].astype('category')

df.info()

#Univariate analysis-Numerical
num_cols=df.select_dtypes(include=np.number).columns
df[num_cols].describe()

sns.histplot(df['age'],bins=30,kde=True,color='skyblue')
plt.title("Age Distribution of Passengers")
plt.show()

sns.boxplot(x=df['fare'],color='orange')
plt.title("Fare Distribution of Passengers")
plt.show()

#Univariate Analysis-Categorical
cat_cols=df.select_dtypes(include='category').columns
df[cat_cols].describe()

for col in cat_cols:
    plt.title(f"Count of: {col}")
    sns.catplot(x=col,data=df)
    plt.show()

#Bivariate Analysis- Numerical vs Categorical
sns.boxplot(x=df['class'],y=df['fare'],color='orange')
plt.title("Fare by Class")
plt.show()

sns.boxplot(x=df['sex'],y=df['age'],color='orange')
plt.title("Age by Gender")
plt.show()

#Bivariate Analysis- Categorical vs Categorical
sns.boxplot(x='survived',hue='sex',data=df)
plt.title("Survival Count by Gender")
plt.show()

sns.boxplot(x='survived',hue='class',data=df)
plt.title("Survival Count by Class")
plt.show()

#Correlation and Multivariate Analysis

corr=df[['age','fare','survived']].corr()
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title("Correlation HeatMap")
plt.show()

sns.pairplot(df[['age','fare','survived']],hue='survived')
plt.show()

#Outlier Detection using IQR

Q1=df['fare'].quantile(0.25)
Q3=df['fare'].quantile(0.75)
IQR=Q3-Q1

lower=Q1-1.5*IQR
upper=Q3 +1.5*IQR

outliers=df[(df['fare']<lower)|(df['fare']>upper)]
print("No. of outliers: ",len(outliers))

#Outliers
df['fare_capped']=df['fare'].clip(lower,upper)
sns.boxplot(x=df['fare_capped'])
plt.title("Fare After Outlier Clipping")
plt.show()