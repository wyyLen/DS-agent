"""
    1. 数据预分析
    2. 复杂数据预处理
    3. 特征工程
    4. 机器学习
    5. 相关性分析
    6. 异常值检测
    7. 统计分析
    8. 分布分析
"""


# 1. Prompt for taking on "pda" tasks
PRE_DA_PROMPT = """
The current task is about pre-analysis data, please note the following:
- Distinguish column types with `select_dtypes` for tailored analysis and visualization, such as correlation.
- Remember to `import numpy as np` before using Numpy functions.
"""

# 2. Prompt for taking on "data_preprocess" tasks
DATA_PREPROCESS_PROMPT = """
The current task is about data preprocessing, please note the following:
- Monitor data types per column, applying appropriate methods.
- Ensure operations are on existing dataset columns.
- Avoid writing processed data to files.
- Avoid any change to label column, such as standardization, etc.
- Prefer alternatives to one-hot encoding for categorical data.
- Only encode or scale necessary columns to allow for potential feature-specific engineering tasks (like time_extract, binning, extraction, etc.) later.
- Each step do data preprocessing to train, must do same for test separately at the same time.
- Always copy the DataFrame before processing it and use the copy to process.
"""

# 3. Prompt for taking on "feature_engineering" tasks
FEATURE_ENGINEERING_PROMPT = """
The current task is about feature engineering. when performing it, please adhere to the following principles:
- Generate as diverse features as possible to improve the model's performance step-by-step. 
- Use available feature engineering tools if they are potential impactful.
- Avoid creating redundant or excessively numerous features in one step.
- Exclude ID columns from feature generation and remove them.
- Each feature engineering operation performed on the train set must also applies to the test separately at the same time.
- Avoid using the label column to create features, except for cat encoding.
- Use the data from previous task result if exist, do not mock or reload data yourself.
- Always copy the DataFrame before processing it and use the copy to process.
"""

# 4. Prompt for taking on "model_train" tasks
MODEL_TRAIN_PROMPT = """
The current task is about training a model, please ensure high performance:
- Keep in mind that your user prioritizes results and is highly focused on model performance. So, when needed, feel free to use models of any complexity to improve effectiveness, such as XGBoost, CatBoost, etc.
- If non-numeric columns exist, perform label encode together with all steps.
- Use the data from previous task result directly, do not mock or reload data yourself.
- Set suitable hyperparameters for the model, make metrics as high as possible.
"""

# 4. Prompt for taking on "model_evaluate" tasks
MODEL_EVALUATE_PROMPT = """
The current task is about evaluating a model, please note the following:
- Ensure that the evaluated data is same processed as the training data. If not, remember use object in 'Done Tasks' to transform the data.
- Use trained model from previous task result directly, do not mock or reload model yourself.
"""

# 4. Prompt for taking on "machine_learning" tasks
MACHINE_LEARNING_PROMPT = """
The current task is about machine learning, please note the following:
In model training:
- Keep in mind that your user prioritizes results and is highly focused on model performance. 
- If non-numeric columns exist, perform label encode together with all steps.
- Use the model from previous tasks result directly, do not mock or reload model yourself.
- Set suitable hyperparameters for the model, make metrics as high as possible.

In model evaluation:
- Ensure that the evaluated data is same processed as the training data. If not, remember use object in 'Done Tasks' to transform the data.
- Use trained model from previous task result directly, do not mock or reload model yourself.
"""

# 5. Prompt for taking on "correlation_analysis" tasks
CORRELATION_ANALYSIS_PROMPT = """
The current task is about correlation analysis, please note the following:
- Use the corr() method in Pandas to compute the correlation matrix.
- Use different methods like Pearson, Spearman, or Kendall as needed.
- Analyze the correlation values to understand the strength and direction of relationships between variables.
"""

# 6. Prompt for taking on "outlier_detection" tasks
OUTLIER_DETECTION_PROMPT = """
The current task is about outlier detection, please note the following:
- Calculate the Z-scores for numerical columns using zscore() from scipy.stats.
- Calculate the first and third quartiles (Q1 and Q3) and the IQR.
"""

# 7. Prompt for taking on "statistical_analysis" tasks
STATISTICAL_ANALYSIS_PROMPT = """
The current task is about statistical analysis, please note the following:
- Calculate the mean, median, mode, standard deviation, variance, minimum, maximum, percentiles, skewness, and kurtosis.
- Store these statistics in a dictionary and convert it to a DataFrame for easy viewing.
"""

# 8. Prompt for taking on "distribution_analysis" tasks
DISTRIBUTION_ANALYSIS_PROMPT = """
The current task is about distribution analysis, please note the following:
- Conduct normality tests such as Shapiro-Wilk, Anderson-Darling, and Kolmogorov-Smirnov tests. Their distribution characteristics are reported upon request.
"""
