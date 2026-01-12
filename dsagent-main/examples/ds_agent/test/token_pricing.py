import matplotlib.pyplot as plt

# 模型名称
models = ['GPT-4o-mini', 'GPT-4o', 'o1-preview', 'Claude-3.5-sonnet', 'GLM-4-Plus']

# 每百万token的API价格（假设这些值，替换成实际数据）
prices_per_1m_tokens = [0.15, 2.50, 15, 3.00, 7.04]

plt.figure(figsize=(10, 6))
bars = plt.bar(models, prices_per_1m_tokens, color=['blue', 'green', 'orange', 'red', 'purple'])

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f'{yval:.2f}', ha='center', va='bottom', fontsize=12)

plt.title('API Pricing Comparison per 1M Tokens', fontsize=22)
plt.xlabel('Model', fontsize=18)
plt.ylabel('Price per 1M Tokens ($)', fontsize=18)

plt.xticks(fontsize=16, rotation=0)
plt.tight_layout()
plt.show()
