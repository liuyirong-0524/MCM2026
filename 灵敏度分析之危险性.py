import numpy as np
import matplotlib.pyplot as plt

# 基础参数
V_death = 2.5        # 电池报废的临界电压 (铜溶解开始)
Self_discharge_rate = 0.0005  # 每天自放电掉 0.05% 的 SOC
Capacity_Ah = 4.2    # 4500mAh

def get_ocv_low(z):
    """专门针对低电量区间的 OCV 拟合"""
    z = np.clip(z, 1e-5, 1.0)
    # 模拟低电量下电压快速崩塌的特性
    return 3.7 + 0.1*z - 0.02/z + 0.05*np.log10(z)

# 灵敏性分析范围：设置不同的截止电压
v_cutoff_range = np.linspace(3.0, 3.5, 50)
days_to_death = []

for vc in v_cutoff_range:
    # 1. 寻找关机时的 SOC (z_start_static)
    # 假设关机前瞬时小电流负载，导致 z 处于较低位置
    z_search = np.linspace(0.0001, 0.2, 5000)
    ocvs = get_ocv_low(z_search)
    idx = np.where(ocvs >= vc)[0]
    z_at_shutdown = z_search[idx[0]] if len(idx) > 0 else 0.0001
    
    # 2. 模拟自放电过程：z_t = z_start - rate * days
    # 寻找 z_death 使得 OCV(z_death) = V_death
    z_death = 0.005 # 假设 SOC 跌到 0.5% 时电压跌破 2.5V
    
    # 3. 计算从关机到报废的天数
    days = (z_at_shutdown - z_death) / Self_discharge_rate
    days_to_death.append(max(0, days))

# --- 可视化 ---
plt.figure(figsize=(10, 6))
plt.plot(v_cutoff_range, days_to_death, color='#2ecc71', lw=3, label='Grace Period before Failure')
plt.fill_between(v_cutoff_range, days_to_death, color='#2ecc71', alpha=0.1)

# 辅助线：通常要求的 15 天安全存放期
plt.axhline(y=15, color='red', linestyle='--', alpha=0.6)
plt.text(3.4, 17, 'Industry Buffer Min (15 Days)', color='red', fontsize=10)

plt.title('Sensitivity: $V_{cutoff}$ vs. Battery Survival Time (Uncharged)', fontsize=14)
plt.xlabel('BMS Cut-off Voltage ($V_{cutoff}$) [V]', fontsize=12)
plt.ylabel('Days until Permanent Damage [Days]', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()