import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 基础模型定义 (OCV)
# ==========================================
def get_ocv_full(z):
    """Combined+3 OCV 模型"""
    K0, K1, K2, K3, K4 = 3.7, 0.2, -0.01, 0.05, -0.02
    z = np.clip(z, 1e-4, 1-1e-4)
    return K0 + K1*z + K2/z + K3*np.log(z) + K4*np.log(1-z)

def get_ocv_low(z):
    """低电量区间的 OCV 拟合 (用于生存时间分析)"""
    z = np.clip(z, 1e-5, 1.0)
    return 3.7 + 0.1*z - 0.02/z + 0.05*np.log10(z)

# ==========================================
# 2. 参数设置与循环计算
# ==========================================
v_cutoff_range = np.linspace(3.0, 3.55, 50)
N_base = 600 
w = 2.2 
Self_discharge_rate = 0.0028
z_death = 0.005 

cycle_lives = []
days_to_death = []

for vc in v_cutoff_range:
    # --- A. 循环寿命计算 ---
    z_space_life = np.linspace(0.0, 1.0, 1000)
    ocv_life = get_ocv_full(z_space_life)
    idx_l = np.where(ocv_life <= vc)[0]
    z_remain_life = z_space_life[idx_l[-1]] if len(idx_l) > 0 else 0
    dod = 1.0 - z_remain_life
    cycles = N_base / (dod**w)
    cycle_lives.append(cycles)
    
    # --- B. 报废生存时间计算 ---
    z_space_death = np.linspace(0.0001, 0.2, 5000)
    ocv_death = get_ocv_low(z_space_death)
    idx_d = np.where(ocv_death >= vc)[0]
    z_at_shutdown = z_space_death[idx_d[0]] if len(idx_d) > 0 else 0.0001
    days = (z_at_shutdown - z_death) / Self_discharge_rate
    days_to_death.append(max(0, days))

# ==========================================
# 3. 可视化整合
# ==========================================
fig, ax1 = plt.subplots(figsize=(11, 7), dpi=100)


color1 = "#57157A80"
ax1.set_xlabel('BMS Cut-off Voltage ($V_{cutoff}$) [V]', fontsize=12,fontweight='bold')
ax1.set_ylabel('Cycle Life (Full Cycles until 80% SOH)', color=color1, fontsize=12,fontweight='bold')
line1 = ax1.plot(v_cutoff_range, cycle_lives, color=color1, lw=3, label='Cycle Life')
ax1.tick_params(axis='y', labelcolor=color1)
ax1.grid(True, linestyle=':', alpha=0.6)

ax2 = ax1.twinx()
color2 = "#106634b3"
ax2.set_ylabel('Survival Time after 0% (Days until Damage)', color=color2, fontsize=12,fontweight='bold')
line2 = ax2.plot(v_cutoff_range, days_to_death, color=color2, lw=3, label='Survival Days')
# 假设 Survival Days 范围在 0-40，我们可以调低上限，让曲线显得陡峭靠上
ax2.set_ylim(5, 45)
# ax2.fill_between(v_cutoff_range, days_to_death, color=color2, alpha=0.1)
ax2.tick_params(axis='y', labelcolor=color2)

# plt.axhline(y=15, color='#e74c3c', linestyle='--', lw=1.5, alpha=0.7)
# plt.text(3.02, 17, 'Industry Min Buffer (15 Days)', color='#e74c3c', fontweight='bold')

lns = line1 + line2
labs = [l.get_label() for l in lns]
ax1.legend(lns, labs, loc='upper left', frameon=True, shadow=True)

plt.title('Sensitivity Analysis: Battery Longevity vs. Safety Buffer', fontsize=14, pad=20)
fig.tight_layout()
plt.show()