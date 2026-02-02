import numpy as np
import matplotlib.pyplot as plt

# 假设参数
Capacity_Ah = 4.2  # 4500mAh 手机电池
I_standby = 0.02  
z_start = 1.0      # 从满电开始待机
# 基础物理参数
V_cutoff = 3.2
I = 1.0  
T_ref_c = 25.0  # 基准温度 (25°C)
T_ref_k = T_ref_c + 273.15
R_base = 0.08
Ea_Rg = 4000 
z_full = 1.0

# Combined+3 OCV 模型
def get_ocv(z):
    K0, K1, K2, K3, K4 = 3.7, 0.2, -0.01, 0.05, -0.02
    # 防止 log(0) 或负数
    z = np.clip(z, 1e-4, 1-1e-4)
    return K0 + K1*z + K2/z + K3*np.log(z) + K4*np.log(1-z)

def get_z_from_ocv(v_target):
    """通过 OCV 反推 SOC (z)"""
    z_space = np.linspace(0.001, 1.0, 5000)
    ocv_values = get_ocv(z_space)
    # 寻找最接近目标电压的 SOC
    idx = np.argmin(np.abs(ocv_values - v_target))
    return z_space[idx]

# 灵敏性分析范围
v_cutoff_range = np.linspace(3.0, 3.6, 50)
standby_hours = []

for vc in v_cutoff_range:
    z_end = get_z_from_ocv(vc)
    usable_soc = max(0, z_start - z_end)
    hours = (usable_soc * Capacity_Ah) / I_standby
    standby_hours.append(hours)

# 计算变化率 (Sensitivity)
sensitivity = -np.diff(standby_hours) / np.diff(v_cutoff_range)

# 可视化
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Cut-off Voltage (V)')
ax1.set_ylabel('Standby Time (Hours)', color=color)
ax1.plot(v_cutoff_range, standby_hours, color=color, lw=2, label='Standby Duration')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

# ax2 = ax1.twinx()  # 开启第二个 Y 轴显示灵敏度
# color = 'tab:red'
# ax2.set_ylabel('Sensitivity (Hours lost per 0.1V)', color=color)
# ax2.plot(v_cutoff_range[:-1], sensitivity/10, '--', color=color, alpha=0.7, label='Sensitivity')
# ax2.tick_params(axis='y', labelcolor=color)

plt.title('Standby Time Sensitivity to Cut-off Voltage')
fig.tight_layout()
plt.show()