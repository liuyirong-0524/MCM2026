import numpy as np
import matplotlib.pyplot as plt

class FinalThrottlingSim:
    def __init__(self):
        self.Q_nom = 4500 * 3600 / 1000 
        self.V_cutoff = 3.0           
        self.R_s = 0.15  
        self.z_full = 1.0                
        self.K = [3.8, 0.15, -0.01, 0.08, -0.01] 
        self.k1, self.k2 = 0.05, 10.0

    def get_OCV(self, z):
        z = np.clip(z, 0.0001, 0.9999)
        return self.K[0] + self.K[1]*z + self.K[2]/z + \
               self.K[3]*np.log(z) + self.K[4]*np.log(1-z)

    def get_beta(self, s, enabled=True):
        if not enabled: return 1.0  
        if s > 20: return 1.0  
        if s > 10: return 0.8  
        if s > 1:  return 0.5 
        return 0.1 

    def run(self, p_base, use_throttling=True):
        z = self.z_full
        t = 0
        dt = 5 
        res = {'t_min':[], 'soc':[]}
        while z > 0:
            s = (z / self.z_full) * 100
            beta = self.get_beta(s, enabled=use_throttling)
            p_load = (p_base + np.random.normal(0, 0.01)) * beta
            v_ocv = self.get_OCV(z)
            r = self.R_s * (1 + self.k1 * np.exp(-self.k2 * z))
            delta = v_ocv**2 - 4 * r * p_load
            res['t_min'].append(t/60)
            res['soc'].append(s)
            if delta < 0 or s <= 0.01:
                break
            i = (v_ocv - np.sqrt(delta)) / (2 * r)
            z -= (i * dt) / self.Q_nom
            t += dt
        
        res['t_min'].append(t/60)
        res['soc'].append(0)
        return res

# --- 绘图完善 ---
sim = FinalThrottlingSim()
p_bases = [0.80, 4.25, 1.48, 1.40, 1.12, 2.35] 
colors = ['#1abc9c', '#3498db', '#9b59b6', '#e74c3c', "#d30082", "#c0b42b"]
labels = ['Call', 'Camera', 'GPS', 'Social', 'Video', 'Game']

plt.figure(figsize=(14, 8), dpi=100)
max_t = 0

for pb, col, lab in zip(p_bases, colors, labels):
    # 获取数据
    d_thr = sim.run(pb, True)
    d_raw = sim.run(pb, False)
    
    # 记录最大时间
    max_t = max(max_t, d_thr['t_min'][-1])
    
    # 1. 绘制无节流虚线 (从SOC=20开始)
    s_raw = np.array(d_raw['soc'])
    idx_20 = np.where(s_raw <= 20)[0][0]
    plt.plot(d_raw['t_min'][idx_20:], d_raw['soc'][idx_20:], 
             color=col, linestyle='--', linewidth=2.0, alpha=0.6)
    
    # 2. 绘制有节流实线
    plt.plot(d_thr['t_min'], d_thr['soc'], color=col, linewidth=2.5, label=lab)
    
plt.axhline(y=0, color='black', linewidth=1.5) 
plt.axhline(y=20, color='gray', linestyle=':', alpha=0.3)

plt.title('Battery SOC Decay', fontsize=14, pad=20)
plt.xlabel('Time (Minutes)', fontsize=12)
plt.ylabel('SOC (%)', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.4)
plt.ylim(-5, 105) 
plt.xlim(0, max_t + 50)

# 图例
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color='black', lw=2),
                Line2D([0], [0], color='black', lw=1.2, linestyle='--')]
leg1 = plt.legend(custom_lines, ['Throttled ', 'Normal '], loc='center right')
plt.gca().add_artist(leg1)
plt.legend(loc='upper right', ncol=2)

plt.tight_layout()
plt.show()