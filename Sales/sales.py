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

