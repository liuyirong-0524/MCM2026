import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

V_cutoff = 3.2
I = 1.0  
T_ref_c = 25.0  # 基准温度 (25°C)
T_ref_k = T_ref_c + 273.15
R_base = 0.08
Ea_Rg = 4000 
z_full = 1.0
# 1. 基础物理与 SOC-OCV 模型 (沿用你的 Combined+3 模型)
def get_ocv(z):
    K0, K1, K2, K3, K4 = 3.7, 0.2, -0.01, 0.05, -0.02
    z = np.clip(z, 1e-4, 1-1e-4)
    return K0 + K1*z + K2/z + K3*np.log(z) + K4*np.log(1-z)

def calculate_z_cutoff_custom(temp_c, v_cut, current=1.0):
    T_k = temp_c + 273.15
    R = R_base * np.exp(Ea_Rg * (1/T_k - 1/T_ref_k))
    z_search = np.linspace(0.0001, 1.0, 2000)
    v_term = get_ocv(z_search) - current * R
    idx = np.where(v_term >= v_cut)[0]
    return z_search[idx[0]] if len(idx) > 0 else 1.0

# 参数设定
v_range = np.linspace(3.0, 3.5, 30)
I_standby = 0.02  # 10mA 待机电流
Cap_Ah = 4.2      # 4500mAh
phi_threshold = 0.8 # 可承受低温标准：容量保持 80%

standby_times = []
min_work_temps = []

# 2. 核心计算循环
for vc in v_range:
    # A. 计算待机时间 (假设 25°C 静态环境)
    z_end_standby = calculate_z_cutoff_custom(25.0, vc, current=I_standby)
    hours = (Cap_Ah * (1.0 - z_end_standby)) / I_standby
    standby_times.append(hours)
    
    # B. 计算最低可承受温度 (Phi = 0.8)
    def phi_gap(t):
        z_ref = calculate_z_cutoff_custom(25.0, vc, current=1.0) # 使用 1A 负载作为工作基准
        z_curr = calculate_z_cutoff_custom(t, vc, current=1.0)
        phi = (1.0 - z_curr) / (1.0 - z_ref) if (1.0 - z_ref) > 0 else 0
        return phi - phi_threshold
    
    t_min = fsolve(phi_gap, x0=0.0)[0]
    min_work_temps.append(t_min)

# 3. 可视化绘制
fig, ax1 = plt.subplots(figsize=(11, 7), dpi=100)

# 绘制待机时间 (左轴)
color1 = '#2c3e50'
ax1.set_xlabel('Cut-off Voltage ($V_{cutoff}$) [V]', fontsize=12,fontweight='bold')
ax1.set_ylabel('Standby Duration [Hours]', color=color1, fontsize=12,fontweight='bold')
line1 = ax1.plot(v_range, standby_times, color=color1, lw=3, label='Standby Time (Room Temp)')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle='--', alpha=0.5)

# 绘制最低温度 (右轴)
ax2 = ax1.twinx()
color2 = '#e67e22'
ax2.set_ylabel('Min Operable Temperature [$^\circ$C]', color=color2, fontsize=12)
line2 = ax2.plot(v_range, min_work_temps, color=color2, lw=3, ls='--', label='Min Working Temp (80% Cap)')
ax2.tick_params(axis='y', labelcolor=color2)
ax2.invert_yaxis() # 将低温放在下方更直观

# 合并图例
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='upper center', frameon=True, shadow=True)

plt.title(' Standby Time vs. Low-Temp Resilience', fontsize=14, pad=20)
fig.tight_layout()
plt.show()