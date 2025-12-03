import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("amazon_products_dataset.csv")
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())
print(df['category'].unique())
print("Null values in columns")
print(df.isnull().sum())


print(df.isnull().sum().sum())

for col in df.columns:
  if(df[col].isnull().sum()>0):
    if(df[col].dtype=='object'):
      df[col].fillna("Unknown",inplace=True)
    else:
      df[col].fillna(df[col].median(),inplace=True)

print(df.isnull().sum())

df.drop_duplicates(inplace=True)

numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
df = df.reset_index(drop=True)
df.head()

print(df.sort_values(by='actual_price',ascending=False).head())

print(df.groupby('brand')['discount_price'].mean().sort_values(ascending=False))

print(df[(df['rating']>=4.5) & (df['rating_count']>=2000)])

print(df.groupby('category')['actual_price'].mean())

df['discount_percent'] = ((df['actual_price'] - df['discount_price']) / df['actual_price']) * 100
df.loc[df['actual_price'] < 500, 'price_category'] = 'Budget'
df.loc[(df['actual_price'] >= 500) & (df['actual_price'] < 2000), 'price_category'] = 'Midrange'
df.loc[df['actual_price'] >= 2000, 'price_category'] = 'Premium'
df['popularity_score'] = df['rating'] * np.log(df['rating_count'] + 1)
print(df.sort_values(by='popularity_score',ascending=False).head(10))

print(df[(df['price_category']=='Premium') & (df['discount_percent']>40)])
print(df[df['title'].str.contains('Bluetooth')])
print(df[(df['rating'] >= 4.0) &(df['discount_percent'].between(20, 50)) &(df['popularity_score'] >= df['popularity_score'].quantile(0.7))])

print(df.groupby('brand')['discount_percent'].mean())
print(df.groupby('price_category')['popularity_score'].mean())
print(df.groupby('category')['discount_percent'].mean().sort_values(ascending=False).head(1))
df['revenue_potential'] = df['actual_price'] * df['rating_count']
print(df.groupby('brand')['revenue_potential'].sum())

plt.title("Price distribution")
sns.histplot(df['actual_price'],kde=True)
plt.show()

plt.title("Relationship between price and rating")
sns.scatterplot(data=df,x='actual_price',y='rating')
plt.show()

plt.title("Category-wise average rating")
sns.barplot(data=df,x='category',y='rating')
plt.show()

plt.title("Brand product count")
sns.countplot(data=df,x='brand')
plt.show()

plt.title("Numeric feature correlations")
sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="coolwarm")
plt.show()

#Exporting
df.to_csv("amazon_products_dataset_cleaned.csv")

clean=pd.read_csv("amazon_products_dataset_cleaned.csv")
print(clean.head())