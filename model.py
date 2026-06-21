# Loading the dataset
import pandas as pd
df = pd.read_csv("employee_attrition_train.csv")
df.head()

# dataset shape
df.shape

# dataset info
df.info()

# checking for null values
df.isnull().sum()

# percentage of null values in each column
round(df.isnull().sum()/len(df.index), 4)*100

# Missing values treatment
# Numerical columns
num_cols = [
    "Age",
    "DistanceFromHome",
    "DailyRate"
]

# Fill numerical columns with median
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

# Categorical columns
cat_cols = [
    "BusinessTravel",
    "MaritalStatus"
]

# Fill categorical columns with mode
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\nMissing Values After:")
print(df.isnull().sum())

# Searching for duplicate values
# Number of duplicate rows
duplicates = df.duplicated().sum()

print("Duplicate Rows:", duplicates)

# Outliers treatment
# Creating a copy of dataframe
df_cleaned = df.copy()

# Numerical columns
num_cols = df_cleaned.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove identifier columns if present
cols_to_exclude = ['EmployeeNumber', 'EmployeeCount', 'StandardHours']

num_cols = [col for col in num_cols if col not in cols_to_exclude]

print(num_cols)

# Checking Number of outliers before treatment
for col in num_cols:

    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((df_cleaned[col] < lower) | (df_cleaned[col] > upper)).sum()

    print(f"{col}: {outliers} outliers")

    # Plotting Boxplots before outliers treatment
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_cols = 3
    n_rows = (len(num_cols) + n_cols - 1) // n_cols

    plt.figure(figsize=(18, 5*n_rows))

    for i, col in enumerate(num_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.boxplot(x=df_cleaned[col])
        plt.title(col)

    plt.tight_layout()
    plt.show()

    # Apply IQR Capping
    import numpy as np

for col in num_cols:

    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    # Upper capping
    df_cleaned[col] = np.where(
        df_cleaned[col] > upper,
        upper,
        df_cleaned[col]
    )

    # Lower capping
    df_cleaned[col] = np.where(
        df_cleaned[col] < lower,
        lower,
        df_cleaned[col]
    )

print("Outlier capping completed.")

# Plot Boxplots After Treatment
plt.figure(figsize=(18, 5*n_rows))

for i, col in enumerate(num_cols, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.boxplot(x=df_cleaned[col])
    plt.title(col)

plt.tight_layout()
plt.show()


# checking for outliers after treatment
for col in num_cols:

    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((df_cleaned[col] < lower) | (df_cleaned[col] > upper)).sum()

    print(f"{col}: {outliers} outliers remaining")

    # Descriptive Statistics
    df_cleaned.describe()

    # Skewness Table
    import pandas as pd

    # Numerical columns
    num_cols = df_cleaned.select_dtypes(
        include=['int64', 'float64']
    ).columns

    # Skewness table
    skew_df = pd.DataFrame({
        'Feature': num_cols,
        'Skewness': [df_cleaned[col].skew() for col in num_cols]
    })

    skew_df = skew_df.sort_values(
        by='Skewness',
        ascending=False
    )

    print(skew_df.round(3))

    def skew_category(x):
        if abs(x) < 0.5:
            return "Approximately Normal"
        elif abs(x) < 1:
            return "Moderately Skewed"
        else:
            return "Highly Skewed"

skew_df['Category'] = skew_df['Skewness'].apply(
    skew_category
)

print(skew_df)

plt.figure(figsize=(12,8))

sns.barplot(
    data=skew_df,
    x='Skewness',
    y='Feature'
)

plt.axvline(
    x=0,
    linestyle='--'
)

plt.title("Skewness of Numerical Features")
plt.xlabel("Skewness")
plt.ylabel("Features")

plt.show()

# Kurtosis Table
num_cols = df_cleaned.select_dtypes(
    include=['int64', 'float64']
).columns

kurt_df = pd.DataFrame({
    'Feature': num_cols,
    'Kurtosis': [
        df_cleaned[col].kurt()
        for col in num_cols
    ]
})

kurt_df = kurt_df.sort_values(
    by='Kurtosis',
    ascending=False
)

print(kurt_df.round(3))

def kurtosis_category(x):

    if x < 0:
        return "Platykurtic"

    elif x > 0:
        return "Leptokurtic"

    else:
        return "Mesokurtic"

kurt_df['Category'] = kurt_df[
    'Kurtosis'
].apply(kurtosis_category)

print(kurt_df)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,8))

sns.barplot(
    data=kurt_df,
    x='Kurtosis',
    y='Feature'
)

plt.axvline(
    x=0,
    linestyle='--',
    color='black'
)

plt.title(
    "Kurtosis of Numerical Features"
)

plt.show()


stats_df = pd.DataFrame({
    'Feature': num_cols,
    'Mean': df_cleaned[num_cols].mean().values,
    'Median': df_cleaned[num_cols].median().values,
    'Std Dev': df_cleaned[num_cols].std().values,
    'Skewness': df_cleaned[num_cols].skew().values,
    'Kurtosis': df_cleaned[num_cols].kurt().values
})

print(stats_df.round(3))

# Correlation Heatmap (EDA Phase)
# Creating a copy dataframe
df_cap = df_cleaned.copy()

num_df = df_cap.select_dtypes(
    include=['int64', 'float64']
)

drop_cols = [
    'EmployeeNumber',
    'EmployeeCount',
    'StandardHours'
]

num_df = num_df.drop(
    columns=[col for col in drop_cols if col in num_df.columns]
)

corr_matrix = num_df.corr()

mask = np.triu(
    np.ones_like(corr_matrix, dtype=bool)
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=0.3,
    annot_kws={"size": 8}
)

plt.title(
    "Employee Attrition Dataset Correlation Heatmap",
    fontsize=12
)

plt.xticks(fontsize=8, rotation=45)
plt.yticks(fontsize=8)

plt.tight_layout()
plt.show()

# Feature Engineering (One hot encoding)
# Creating Copy
df_ml = df_cleaned.copy()

# Encode Target Variable
df_ml['Attrition'] = df_ml['Attrition'].map({
    'Yes': 1,
    'No': 0
})

# One Hot Encoding
df_ml = pd.get_dummies(
    df_ml,
    drop_first=True
)

# Correlation with Attrition

corr_attrition = (
    df_ml.corr()['Attrition']
    .sort_values(ascending=False)
)

plt.figure(figsize=(8,12))

sns.heatmap(
    corr_attrition.to_frame(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title(
    "Feature Correlation with Attrition",
    fontsize=16
)

plt.show()

print(df_ml.shape)

# Splitting the dataset into features and target variable
X = df_ml.drop('Attrition', axis=1)

y = df_ml['Attrition']

# Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Baseline Model
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=3000)

lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1 Score :", f1_score(y_test, y_pred))
