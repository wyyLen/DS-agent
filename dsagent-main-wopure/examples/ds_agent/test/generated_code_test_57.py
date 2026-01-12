import pandas as pd

# note: 缺乏步骤： 1.处理diff列，"37,410" 应处理为 int； 2.per_point_diff计算应该取abs；
#  3.计算pearson系数时应该使用 scipy.stats.pearson方法。使用df.corr计算得到两个相同的pearson系数而没有p-value。


file_path = 'D:/Dev/DSAgent/data/di_dataset/da_bench/da-dev-tables/election2016.csv'
df = pd.read_csv(file_path)
# df['diff_votes'] = df['votes_dem'] - df['votes_gop']
df['per_point_diff'] = (df['per_dem'] - df['per_gop']) * 100
print(df.head())

pearson_corr = df['diff'].corr(df['per_point_diff'], method='pearson')
p_value = df['diff'].corr(df['per_point_diff'], method='pearson')
print(pearson_corr, p_value)
print("\n\n\n")


import pandas as pd
import numpy as np

# Load the dataset
file_path = r"D:\Dev\DSAgent\data\di_dataset\da_bench\da-dev-tables\election2016.csv"
data = pd.read_csv(file_path)
numerical_data = data.select_dtypes(include=[np.number])
categorical_data = data.select_dtypes(exclude=[np.number])

data_cleaned = data.copy()
data_cleaned['diff'] = data_cleaned['diff'].str.replace(',', '').astype(int)
data_cleaned['per_point_diff'] = data_cleaned['per_point_diff'].str.rstrip('%').astype(float) / 100

# Display the cleaned data types and head to verify the transformations
data_cleaned['percentage_point_diff'] = abs(data_cleaned['per_dem'] - data_cleaned['per_gop'])
from scipy.stats import pearsonr
correlation, p_value = pearsonr(data_cleaned['diff'], data_cleaned['percentage_point_diff'])
print(f"Pearson correlation coefficient (r): {correlation}")
print(f"P-value: {p_value}")
# print(data_cleaned['diff'].corr(data_cleaned['percentage_point_diff'], method='pearson'))

# Interpretation based on the given constraints
if p_value < 0.05:
    if abs(correlation) >= 0.5:
        print("There is a significant linear relationship.")
    else:
        print("There is a significant nonlinear relationship.")
else:
    print("There is no significant correlation.")
