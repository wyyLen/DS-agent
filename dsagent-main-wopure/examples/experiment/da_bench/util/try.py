import pandas as pd
import os


def remove_whitespace_in_column_names(folder_path):
    # 遍历指定文件夹内的所有文件
    for filename in os.listdir(folder_path):
        # 只处理 .csv 文件
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            # 读取CSV文件
            df = pd.read_csv(file_path)

            # 去除列名前后的空格
            df.columns = df.columns.str.strip()

            # 保存文件，覆盖原文件
            df.to_csv(file_path, index=False)
            print(f'Processed file: {filename}')


path = r"D:\Dev\DSAgent\data\di_dataset\da_bench\da-dev-tables"
remove_whitespace_in_column_names(path)
