import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 参数定义
# ==========================================
# 假设基准：100% DoD (截止电压约 2.8V) 时的循环寿命为 500 次
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

N_base = 600 
w = 2.2  # 寿命衰减指数 (通常在 1.1 到 2.5 之间)

def get_dod_from_vcut(v_cut):
    """根据截止电压推算 DoD (简化线性与平台模型)"""
    # 模拟 SOC-OCV 关系，计算从 100% 放电到 v_cut 所释放的电量比例
    z_space = np.linspace(0.0, 1.0, 1000)
    ocv_values = get_ocv(z_space) # 使用你之前的 Combined+3 模型
    # 找到 v_cut 对应的剩余 SOC
    idx = np.where(ocv_values <= v_cut)[0]
    z_remain = z_space[idx[-1]] if len(idx) > 0 else 0
    return 1.0 - z_remain

# ==========================================
# 2. 灵敏度计算
# ==========================================
v_cutoff_range = np.linspace(3.0, 3.6, 50)
cycle_lives = []
total_energy_delivered = [] # 终生总输出能量

for vc in v_cutoff_range:
    dod = get_dod_from_vcut(vc)
    # 经验寿命公式: Cycle = N_base / (DoD^w)
    cycles = N_base / (dod**w)
    cycle_lives.append(cycles)
    
    # 计算终生总输出电量 = 循环次数 * 每次放电容量
    total_energy_delivered.append(cycles * dod)

# ==========================================
# 3. 可视化
# ==========================================
fig, ax1 = plt.subplots(figsize=(10, 6))

# 曲线 1: 循环次数 (左轴)
color = '#3498db'
ax1.set_xlabel('Cut-off Voltage ($V_{cutoff}$) [V]', fontsize=12)
ax1.set_ylabel('Cycle Life (Times)', color=color, fontsize=12)
ax1.plot(v_cutoff_range, cycle_lives, color=color, lw=3, label='Cycle Life')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle=':', alpha=0.6)

# 曲线 2: 终生总输出能量 (右轴) - 衡量性价比
# ax2 = ax1.twinx()
# color = '#27ae60'
# ax2.set_ylabel('Cumulative Life-span Energy (Relative)', color=color, fontsize=12)
# ax2.plot(v_cutoff_range, total_energy_delivered, '--', color=color, lw=2, label='Total Life Energy')
# ax2.tick_params(axis='y', labelcolor=color)

plt.title('Sensitivity: $V_{cutoff}$ vs. Battery Cycle Life', fontsize=14)
fig.tight_layout()
plt.show()