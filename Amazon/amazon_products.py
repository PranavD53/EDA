'''
Task 1 — Load & Inspect the Dataset
Load the dataset into your notebook.
Display first and last 5 rows.
Print .shape and .info().
Show list of unique product categories.
Identify columns that require cleaning or type conversion.
Task 2 — Clean the Dataset
Identify missing values using .isnull().sum().
Fill missing numeric values (discount_price, rating, etc.) using mean or median.
Fill missing categorical values (brand, category) with "Unknown".
Remove all duplicate rows.
Convert numeric columns to their correct data types.
Reset index after cleaning.
Task 3 — Data Analysis
Identify the top 5 most expensive products (actual_price).
Find the brand with the highest number of products.
Compute the average discount percentage for each brand.
Identify products with rating ≥ 4.5 and rating_count ≥ 2000.
Compute category-wise average price.
Task 4 — Programming Task: Feature Engineering + Logical Filters
Task 4A — Create New Columns
Write code to create:
discount_percent = ((actual_price - discount_price) / actual_price) * 100
price_category based on:
actual_price < 500 → "Budget"
500 ≤ price < 2000 → "Midrange"
price ≥ 2000 → "Premium"
popularity_score = rating * log(rating_count + 1)
Task 4B — Filtering Using New Columns
Using the newly created columns:
Show top 10 highest popularity_score products.
Show all Premium category items with discount_percent > 40%.
Show all products whose title contains "Bluetooth" (string filter).
Show products where:
rating >= 4.0
discount_percent between 20% to 50%
popularity_score in the top 30% percentile
(Use .between() and .quantile())
Task 4C — Grouping with New Columns
Find the average discount_percent for each brand.
Find the average popularity_score for each price_category.
Identify which category offers the highest average discount.
Calculate total revenue potential per brand using:
revenue_potential = actual_price * rating_count
Task 5 — Data Visualization 
Use the specific plot type mentioned:
Price distribution → Histogram + KDE
Relationship between price and rating → Scatter Plot
Category-wise average rating → Bar Plot
Brand product count → Count Plot
Numeric feature correlations → Heatmap
'''
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