import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
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
# ... (保留你之前的 OCV 和基础参数) ...
V_cutoff_range = np.linspace(3.0, 3.4, 20)  # 测试不同的截止电压
phi_threshold = 0.8  # 定义最低可承受标准：容量保持率 80%

def get_phi(temp_c, v_cut):
    """计算特定温度和截止电压下的容量保持率 Phi"""
    # 计算基准状态 (25°C, 当前 v_cut)
    z_ref = calculate_z_cutoff_custom(25.0, v_cut)
    delta_z_ref = z_full - z_ref
    
    # 计算当前状态
    z_curr = calculate_z_cutoff_custom(temp_c, v_cut)
    delta_z_curr = max(0, z_full - z_curr)
    
    return delta_z_curr / delta_z_ref if delta_z_ref > 0 else 0

def calculate_z_cutoff_custom(temp_c, v_cut):
    T_k = temp_c + 273.15
    R = R_base * np.exp(Ea_Rg * (1/T_k - 1/T_ref_k))
    z_search = np.linspace(0.0001, 1.0, 1000)
    v_term = get_ocv(z_search) - I * R
    idx = np.where(v_term >= v_cut)[0]
    return z_search[idx[0]] if len(idx) > 0 else 1.0

# 灵敏性分析：寻找每个 V_cutoff 对应的最低工作温度
min_temps = []
for vc in V_cutoff_range:
    # 使用 fsolve 寻找使得 get_phi - 0.8 = 0 的温度
    func = lambda t: get_phi(t, vc) - phi_threshold
    # 从 0°C 开始向下搜索
    t_min = fsolve(func, x0=0.0)[0]
    min_temps.append(t_min)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(V_cutoff_range, min_temps, 'o-', color='#2980b9', lw=2, label='Min Operating Temp (80% Capacity)')
plt.fill_between(V_cutoff_range, min_temps, -30, color='#2980b9', alpha=0.1)

plt.title('Sensitivity Analysis: $V_{cutoff}$ vs. Minimum Operable Temperature', fontsize=14)
plt.xlabel('Cut-off Voltage ($V_{cutoff}$) [V]', fontsize=12)
plt.ylabel('Minimum Temperature [$^\circ$C]', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.gca().invert_yaxis() # 温度越低越在下方，符合直觉
plt.show()