import numpy as np
import matplotlib.pyplot as plt

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

# 计算特定温度下的 z_cutoff
def calculate_z_cutoff(temp_c):
    T_k = temp_c + 273.15
    R = R_base * np.exp(Ea_Rg * (1/T_k - 1/T_ref_k))
    
    z_search = np.linspace(0.001, 1.0, 1000)
    v_term = get_ocv(z_search) - I * R
    
    # 寻找第一个大于 V_cutoff 的 z 值
    idx = np.where(v_term >= V_cutoff)[0]
    return z_search[idx[0]] if len(idx) > 0 else 1.0

# 1. 计算基准状态下的可用 SOC 区间
z_cutoff_ref = calculate_z_cutoff(T_ref_c)
delta_z_ref = z_full - z_cutoff_ref

# 2. 模拟温度下降过程
delta_temp = np.linspace(0, 50, 100)  # 温度下降量从 0 到 50 度
phi_list = []

for dt in delta_temp:
    current_temp = T_ref_c - dt
    z_cutoff_curr = calculate_z_cutoff(current_temp)
    delta_z_curr = max(0, z_full - z_cutoff_curr)
    
    # 计算比例: 当前可用区间 / 基准可用区间
    phi = delta_z_curr / delta_z_ref
    phi_list.append(phi)

# 3. 可视化
plt.figure(figsize=(9, 6), dpi=100)
plt.plot(delta_temp, np.array(phi_list), color='#e74c3c', lw=2.5, label='Available Capacity Ratio')
plt.fill_between(delta_temp, phi_list, color='#e74c3c', alpha=0.1)

# 修饰图表
plt.title('Capacity Degradation vs. Temperature Drop (ΔT)', fontsize=14)
plt.xlabel('Temperature Drop ΔT [°C] (from 25°C)', fontsize=12)
plt.ylabel('Relative Available Capacity Ratio ($\Phi$)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(1.0, color='black', lw=1, ls='--')
plt.annotate('Baseline ($T_{ref}$)', xy=(0, 1), xytext=(5, 1.05),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1))

# 设置 y 轴上限略高于 1 以便观察
plt.ylim(0, 1.1)
plt.xlim(0, 50)

plt.tight_layout()
plt.show()