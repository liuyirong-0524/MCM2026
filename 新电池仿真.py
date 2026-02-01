import numpy as np
import matplotlib.pyplot as plt

class FinalThrottlingSim:
    def __init__(self):
        # 物理参数
        self.Q_nom = 4500 * 3600 / 1000 
        self.V_cutoff = 3.4             
        self.R_s = 0.15  # 略微调高内阻，使低电量效应更明显
        self.z_full = 1.0               
        self.z_cutoff = 0.045 
        self.K = [3.8, 0.15, -0.05, 0.08, -0.02] 
        self.k1, self.k2 = 0.05, 10.0

    def get_OCV(self, z):
        z = np.clip(z, 0.001, 0.999)
        return self.K[0] + self.K[1]*z + self.K[2]/z + \
               self.K[3]*np.log(z) + self.K[4]*np.log(1-z)

    def get_R_total(self, z):
        return self.R_s * (1 + self.k1 * np.exp(-self.k2 * z))

    def get_beta(self, s, z, p_req):
        if s > 20: return 1.0
        if 10 < s <= 20: return 0.8
        if 1 < s <= 10: return 0.5
        # 1% 电压反馈维持
        v_ocv = self.get_OCV(z)
        r = self.get_R_total(z)
        v_target = self.V_cutoff + 0.05
        i_max = max(0, (v_ocv - v_target) / r)
        f_feedback = (v_target * i_max) / max(0.1, p_req)
        return max(0.05, min(0.3, f_feedback))

    def run(self, p_base=2.0):
        z = self.z_full
        t = 0
        dt = 1 # 秒
        res = {'t_min':[], 'soc':[], 'p_actual':[]}
        
        while z > 0:
            s = (z - self.z_cutoff) / (self.z_full - self.z_cutoff) * 100
            p_req = p_base + np.random.normal(0, 0.1)
            beta = self.get_beta(s, z, p_req)
            p_load = p_req * beta
            
            v_ocv = self.get_OCV(z)
            r = self.get_R_total(z)
            delta = v_ocv**2 - 4 * r * p_load
            if delta < 0: break
            
            i = (v_ocv - np.sqrt(delta)) / (2 * r)
            v_term = v_ocv - i * r
            if v_term < self.V_cutoff: break
            
            res['t_min'].append(t/60)
            res['soc'].append(max(0, s))
            res['p_actual'].append(p_load)
            
            z -= (i * dt) / self.Q_nom
            t += dt
            if s <= 0: break
        return res

# 运行仿真
sim = FinalThrottlingSim()
data = sim.run(p_base=2.2) # 设定一个中等偏高的初始负载

# --- 绘图 ---
plt.figure(figsize=(10, 6), dpi=100)
plt.plot(data['t_min'], data['soc'], color='#2c3e50', linewidth=2.5, label='Display SOC')

# 绘制不同模式的背景色块
plt.axvspan(0, data['t_min'][np.argmin(np.abs(np.array(data['soc'])-20))], color='green', alpha=0.1, label='Normal Mode')
plt.axvspan(data['t_min'][np.argmin(np.abs(np.array(data['soc'])-20))], 
            data['t_min'][np.argmin(np.abs(np.array(data['soc'])-10))], color='orange', alpha=0.1, label='Power Saving')
plt.axvspan(data['t_min'][np.argmin(np.abs(np.array(data['soc'])-10))], 
            data['t_min'][np.argmin(np.abs(np.array(data['soc'])-1))], color='red', alpha=0.1, label='Deep Throttling')
plt.axvspan(data['t_min'][np.argmin(np.abs(np.array(data['soc'])-1))], 
            data['t_min'][-1], color='purple', alpha=0.1, label='Survival Mode')

plt.title('Battery SOC Decay with Perception-Based Throttling', fontsize=14)
plt.xlabel('Time (Minutes)', fontsize=12)
plt.ylabel('Display SOC (%)', fontsize=12)
plt.ylim(0, 100)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

# 关键转折点标注
plt.annotate('20% Throttling', xy=(data['t_min'][np.argmin(np.abs(np.array(data['soc'])-20))], 20), 
             xytext=(+20, +20), textcoords='offset points', arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.show()