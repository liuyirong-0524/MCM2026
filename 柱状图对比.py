import matplotlib.pyplot as plt
import numpy as np

# --- 数据准备 ---
labels = ['Call', 'Camera', 'GPS', 'Social', 'Video', 'Game']

# 模拟数据 (单位: Minutes)
normal_sim = [1250.75, 226.83, 670.58, 709.75, 889.42, 418.33]
throttled_sim = [1416.92, 257.33, 760.08, 804.58, 1008.83, 474.50]

# 官网实测数据 
official_h = [24, 3.5, 12.4, 13.3, 17.5, 7.6]
official_min = [h * 60 for h in official_h]

# --- 绘图设置 ---
x = np.arange(len(labels))  # 标签位置
width = 0.25  # 柱状图宽度

fig, ax = plt.subplots(figsize=(12, 7), dpi=100)

# 绘制三组柱状图
rects1 = ax.bar(x - width, normal_sim, width, label='Sim: Normal', color='#95a5a6', alpha=0.8)
rects2 = ax.bar(x, throttled_sim, width, label='Sim: Throttled', color='#3498db', alpha=0.9)
rects3 = ax.bar(x + width, official_min, width, label='Official: Actual', color='#1abc9c')

# --- 装饰图表 ---
ax.set_ylabel('Time (Minutes)', fontsize=12)
ax.set_title('Battery Life Comparison: Simulation vs. Official Data', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

# 添加数据标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3点纵向偏移
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# 增加网格线增强可读性
ax.yaxis.grid(True, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()
p_bases = [0.80, 4.25, 1.48, 1.40, 1.12, 2.35] 