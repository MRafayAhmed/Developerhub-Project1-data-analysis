import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

# -------------------------------
# 1. Load the Dataset
# -------------------------------
path = "C:/Users/Ansar-PC/Desktop/developerhub-data-analysis/titanic_dataset.csv"
df = pd.read_csv(path)

print("Initial Data Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# -------------------------------
# 2. Data Cleaning
# -------------------------------

# Check missing values
print("\nMissing values before handling:\n", df.isnull().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)
print("\nShape after removing duplicates:", df.shape)

# Separate numerical and categorical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include='object').columns

# Impute numeric columns with median
num_imputer = SimpleImputer(strategy='median')
df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

# Impute categorical columns with most frequent (mode)
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

print("\nMissing values after handling:\n", df.isnull().sum())

# --- Outlier Detection and Management (Using IQR method) ---
def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df

df = remove_outliers(df, numeric_cols)
print("\nShape after outlier removal:", df.shape)

# -------------------------------
# 3. Visualizations
# -------------------------------
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# --- Bar Charts for Categorical Variables ---
for col in categorical_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, data=df)
    plt.title(f'Bar Chart of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# --- Histograms for Numeric Variables ---
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f'Histogram of {col}')
    plt.tight_layout()
    plt.show()

# --- Correlation Heatmap ---
plt.figure(figsize=(10, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Summary of Insights
# -------------------------------

"""
Summary of Insights:
1. The dataset initially contained missing values and duplicate entries that were cleaned.
2. Missing values in numerical columns were imputed using the median; categorical ones with the most frequent value.
3. Duplicates were removed, and outliers were handled using the IQR method.
4. Bar charts revealed the distribution of categorical variables, such as 'sex' and 'embarked'.
5. Histograms showed the spread and skewness of numerical data like 'age' and 'fare'.
6. The correlation heatmap highlighted strong and weak relationships among numeric features.
"""

# Optionally save cleaned dataset
# df.to_csv("cleaned_titanic.csv", index=False)