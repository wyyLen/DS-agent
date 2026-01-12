import subprocess
import pandas as pd
import io

# 获取竞赛列表
output = subprocess.run(['kaggle', 'competitions', 'list', '--csv'], capture_output=True, text=True)

# 将输出转换为 DataFrame
competitions = pd.read_csv(io.StringIO(output.stdout))

# 将 'Deadline' 字段转换为 datetime 类型
competitions['deadline'] = pd.to_datetime(competitions['deadline'], errors='coerce')

# 筛选出已结束的竞赛 (Deadline 在当前时间之前)
ended_competitions = competitions[competitions['deadline'] < pd.Timestamp.now()]

# 打印或保存这些竞赛信息
print(ended_competitions)
