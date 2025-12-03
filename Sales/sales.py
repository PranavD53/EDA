'''
Task 1 — Data Loading, Merging & Initial Inspection
Load all required CSV files into your notebook.
Merge them into one dataset using Store, Dept, and Date.
Display the first and last 10 rows.
Print:
.shape
.info()
.describe()
Identify:
Numerical columns
Categorical columns
Date columns
List all unique store types and departments.
7. Identify which columns may require cleaning or type conversion
Task 2 — Data Cleaning
Identify missing values using .isnull().sum().
Fill missing numeric values (Temperature, Fuel_Price, CPI, Unemployment) using median.
Fill missing markdown-related fields with mean values.
Convert Date column to datetime format.
Remove duplicate rows.
Reset the index after cleaning.
 
Task 3 — Outlier Detection & Treatment
Detect outliers in:
Weekly_Sales
Temperature
Fuel_Price
CPI
Use:
Boxplots
IQR method
Identify if extreme sales spikes occur during holiday weeks.
Decide which outliers should be:
Removed
Capped
Kept as business outliers
Task 4 — Univariate Analysis
Perform univariate analysis (one variable at a time):
Weekly sales distribution (Histogram + KDE).
Store type distribution (Count plot).
Distribution of Temperature, Fuel Price, CPI, Unemployment.
Distribution of sales during:
Holiday weeks
Non-holiday weeks
Identify top 10 departments by average weekly sales.
✅ Task 5 — Bivariate Analysis
Study the relationship between two variables:
Relationship between Temperature & Weekly Sales (scatter plot).
Relationship between Fuel Price & Weekly Sales.
Weekly Sales vs. Store Type.
Weekly Sales vs. Holiday_Flag.
Compare sales between:
Top-performing store
Lowest-performing store
✅ Task 6 — Multivariate Analysis
Analyze more than two variables together:
Create a correlation heatmap for all numeric features.
Analyze store-level sales using:
Store Type
Store Size
Weekly Sales
Multivariate relationship:
Weekly Sales vs Temperature vs Holiday_Flag (3-variable plot or grouped summary)
Analyze whether discount markdowns influence sales when considering:
Date
Holiday weeks
Markdown values
✅ Task 7 — Time Series Analysis
Convert Date to:
Year
Month
Week
Plot total weekly sales over time.
Plot monthly sales trends for:
Store with highest sales
Store with lowest sales
Identify seasonal patterns:
Which months show peak sales?
Which departments show seasonal demand?
🔥 Task 8 — Feature Engineering
Task 8A — Create New Columns
year, month, week → from date.
discount_effect = MarkDown1 + MarkDown2 + MarkDown3 + MarkDown4 + MarkDown5
is_peak_season → True if month in {11, 12}.
normalized_sales = Weekly_Sales / Size
Task 8B — Filter Using Created Columns
Show all peak-season transactions where weekly_sales > 50,000.
Show all stores whose normalized_sales is in the top 10% percentile.
Show departments where discount_effect > median discount.
Filter rows where:
Temperature < 40
Fuel price > 3.5
Weekly sales between 20,000 and 60,000
is_peak_season = True
Task 8C — Grouping & Business Insights
Monthly average sales per store.
Total discount_effect per department.
Department with highest normalized_sales.
Compute store-wise revenue potential:
revenue_potential = Weekly_Sales * 52
Identify top 10 stores based on revenue potential.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
features=pd.read_csv('features.csv')
stores=pd.read_csv('stores.csv')
df=pd.merge(features,stores,on=['Store'])
print(df.head(10))
print(df.tail(10))
print(df.shape)
print(df.info())
print(df.describe())

num_cols=df.select_dtypes(include=np.number).columns
cat_cols=df.select_dtypes(include='object').columns
date_cols=['Date']

print(num_cols)
print(cat_cols)
print(date_cols)

print(df.isnull().sum())
for col in df.columns:
  if df.isnull().sum()[col] > 0:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
      df[col].fillna(df[col].median(), inplace=True)
print(df.isnull().sum())

df['Date']=pd.to_datetime(df['Date'])
print("Duplicates: ",df.duplicated().sum())
df.drop_duplicates()
df.reset_index(inplace=True)

#Outlier Detection and Removal
variables = ['Temperature', 'Fuel_Price', 'CPI']

plt.figure(figsize=(14,6))
for i, var in enumerate(variables, 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x=df[var])
    plt.title(f'Boxplot of {var}')
plt.tight_layout()
plt.show()

def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    return df[(df[column] < lower) | (df[column] > upper)], lower, upper

for col in variables:
    outliers, lower, upper = detect_outliers_iqr(df, col)
    print(f"{col}: {len(outliers)} outliers | Lower={lower:.2f}, Upper={upper:.2f}")

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # Keep only rows within the valid range
    return df[(df[column] >= lower) & (df[column] <= upper)]

cols_to_clean = ['Temperature', 'Fuel_Price', 'CPI']

cleaned_df = df.copy()

for col in cols_to_clean:
    before = len(cleaned_df)
    cleaned_df = remove_outliers_iqr(cleaned_df, col)
    after = len(cleaned_df)
    print(f"{col}: removed {before - after} outliers")

plt.title("Store Type Distribution")
sns.countplot(x=df['Type'])
plt.show()

plt.title("Temperature Distribution")
sns.histplot(df['Temperature'], kde=True)
plt.show()

plt.title("Fuel Price Distribution")
sns.histplot(df['Fuel_Price'], kde=True)
plt.show()

plt.title("CPI Distribution")
sns.histplot(df['CPI'], kde=True)
plt.show()

plt.title("Unemployment Distribution")
sns.histplot(df['Unemployment'], kde=True)
plt.show()

#TASK 5 INCOMPLETE AS SALES CSV FILE IS NOT GIVEN

plt.title("Correlation Heatmap")
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

df['Date'] = pd.to_datetime(df['Date'])

df['Year']  = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week']  = df['Date'].dt.isocalendar().week

md_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
df['discount_effect'] = df[md_cols].sum(axis=1)

df['is_peak_season'] = df['Month'].isin([11, 12])

filtered_rows = df[
    (df['Temperature'] < 40) &
    (df['Fuel_Price'] > 3.5) &
    (df['is_peak_season'])
]
filtered_rows

df['revenue_potential'] = df['Weekly_Sales'] * 52

#8C not possible due to weekly sales column missing