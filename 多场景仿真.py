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

    def get_beta(self, s):
        if s > 20: return 1.0  
        if s > 10: return 0.8  
        if s > 1:  return 0.5 
        return 0.1 

    def run(self, p_base):
        z = self.z_full
        t = 0
        dt = 5 
        res = {'t_min':[], 'soc':[]}
        
        while z > 0:
            s = (z / self.z_full) * 100
            beta = self.get_beta(s)
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

# --- 运行多条曲线 ---
sim = FinalThrottlingSim()
p_bases = [0.83, 4.20, 1.44, 1.40,1.02,2.10]  
colors = ['#1abc9c', '#3498db', '#9b59b6', '#e74c3c',"#d30082", "#c0b42b"]
labels = ['Call', 'Camera', 'GPS', 'Social', 'Video', 'Game']

plt.figure(figsize=(12, 7), dpi=100)

max_time = 0

for pb, col, lab in zip(p_bases, colors, labels):
    data = sim.run(p_base=pb)
    t_ax = np.array(data['t_min'])
    s_ax = np.array(data['soc'])
    
    
    if t_ax[-1] > max_time: max_time = t_ax[-1]
    
    # 绘图
    plt.plot(t_ax, s_ax, color=col, linewidth=2.5, label=lab)
    
    
    idx_1 = np.argmin(np.abs(s_ax - 1))
    plt.scatter(t_ax[idx_1], s_ax[idx_1], color=col, s=20, zorder=6)

plt.axhline(y=20, color='gray', linestyle='--', alpha=0.3)
plt.axhline(y=10, color='gray', linestyle='--', alpha=0.3)
plt.axhline(y=1, color='red', linestyle=':', alpha=0.3)

plt.title('Battery SOC Decay Comparison ', fontsize=14)
plt.xlabel('Time (Minutes)')
plt.ylabel('SOC (%)')
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()
plt.ylim(0, 105)
plt.xlim(0, max_time)


plt.tight_layout()
plt.show()