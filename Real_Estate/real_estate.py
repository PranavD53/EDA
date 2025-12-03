import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Bengaluru_House_Data.csv")
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df.dtypes)

print("Missing values in each column:")
print(df.isnull().sum())
print("Total missing values")
print(df.isnull().sum().sum())

print("Dropping irrelevant columns for price analysis")
df.drop(['society'],axis=1,inplace=True)
print(df.head())

df['bath'].fillna(df['bath'].median(),inplace=True)
df['balcony'].fillna(0,inplace=True)
# total_sqft cleanup
def convert_sqft(x):
  if isinstance(x,str) and '-' in x:
    a,b = x.split('-')
    return (float(a)+float(b))/2
  try:
    return float(x)
  except:
    return None

df['total_sqft']=df['total_sqft'].apply(convert_sqft)
df.dropna(subset=['total_sqft'],inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True,inplace=True)
df.head()

print("Unique Locations: ")
print(df['location'].nunique(),"\n")

print("Average house price for each location")
print(df.groupby('location')['price'].mean().head(),"\n")

print("Location with highest average house price")
print(df.groupby('location')['price'].mean().idxmax(),"\n")

print("Correlation :")
print(df[['total_sqft','bath','price']].corr())

corr_sqft_price = df[['total_sqft','bath','price']].corr().loc['total_sqft', 'price']
 
if corr_sqft_price > 0.7:
    print("• Strong positive correlation → Larger houses generally have higher prices.")
elif corr_sqft_price > 0.4:
    print("• Moderate correlation → Larger houses often cost more, but not always.")
else:
    print("• Weak correlation → Size alone does NOT determine price.")

import seaborn as sns
import matplotlib.pyplot as plt

plt.title("Price Distribution")
sns.histplot(df['price'],kde=True)
plt.show()

plt.title("Relationship between area and price")
sns.scatterplot(x='total_sqft',y='price',data=df)
plt.show()

plt.title("Effect of Bathrooms on Price")
sns.boxplot(x='bath',y='price',data=df)
plt.show()

plt.title("Top 10 Most Expensive Locations")
sns.barplot(df.groupby('location')['price'].mean().sort_values(ascending=False).head(10))
plt.show()


plt.title("Correlation between Numeric Columns")
sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.show()

#Exporting dataset
df.to_csv("Cleaned_Real_Estate.csv")

clean=pd.read_csv("Cleaned_Real_Estate.csv")
clean.head()