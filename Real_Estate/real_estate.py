'''
Task 1 — Load & Inspect the Dataset
1.	Load the dataset  into your notebook.
2.	Display the first 5 and last 5 rows of the dataset.
3.	Print dataset shape (rows, columns).
4.	Print dataset information using .info().
5.	Identify basic data types of all columns.
Task 2 — Clean the Dataset
Identify and count missing values in each column.
Drop column(s) that are irrelevant for price analysis (example: society or unnamed columns).
Handle missing values in numeric columns such as bath, balcony, etc. (use either dropna() or fillna() depending on your reasoning).
Convert total_sqft to numeric — handle values like "2100 - 2850" by converting them to an average or a single number.
Remove duplicate rows.
6. Reset the DataFrame index after cleaning
Task 3 — Data Analysis
Identify the number of unique locations in the dataset.
Compute the average house price for each location.
Find the location with the highest average house price.
Analyze the correlation between numeric columns such as total_sqft, bath, and price. Comment on any strong correlations you find.
Task 4 — Data Visualization
Use the specific visualization types mentioned for each question.
Price Distribution:
Plot the distribution of the price column using a Histogram + KDE curve.
Relationship Between Area and Price:
Visualize the relationship between total_sqft and price using a Scatter Plot.
Effect of Bathrooms on Price:
Show how bath count affects house prices using a Box Plot.
Top 10 Most Expensive Locations:
Plot the Top 10 locations with highest average price using a Bar Chart.
Correlation Between Numeric Columns:
Create a Heatmap to visualize correlations among numeric features (price, total_sqft, bath, balcony, etc.).
'''
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