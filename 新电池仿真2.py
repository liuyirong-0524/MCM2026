import numpy as np
import matplotlib.pyplot as plt

class FinalThrottlingSim:
    def __init__(self):
        # -------- 物理参数 --------
        self.Q_nom = 4500 * 3600 / 1000   # Ah → As
        self.V_cutoff = 3.0              
        self.R_s = 0.15
        self.z_full = 1.0
        self.z_cutoff = 0.0              
        
        # OCV 拟合
        self.K = [3.8, 0.15, -0.05, 0.08, -0.02]
        
        # 内阻 SOC 相关
        self.k1, self.k2 = 0.05, 10.0

    # -------- 模型函数 --------
    def get_OCV(self, z):
        z = np.clip(z, 1e-4, 0.999)
        return (self.K[0] +
                self.K[1]*z +
                self.K[2]/z +
                self.K[3]*np.log(z) +
                self.K[4]*np.log(1-z))

    def get_R_total(self, z):
        return self.R_s * (1 + self.k1 * np.exp(-self.k2 * z))

    def get_beta(self, soc):
        """功率退化（偏物理，不是 UI）"""
        if soc > 20:
            return 1.0
        elif soc > 10:
            return 0.7
        elif soc > 1:
            return 0.4
        else:
            return 0.15  
    
    def run(self, p_base=2.2):
        z = self.z_full
        t = 0
        dt = 1

        res = {
            't_min': [],
            'soc': [],
            'p_actual': [],
            'v_term': []
        }

        while z > 0:
            soc = z * 100
            beta = self.get_beta(soc)

            # 基础负载扰动
            p_req = p_base + np.random.normal(0, 0.05)
            p_load = max(0.01, p_req * beta)

            v_ocv = self.get_OCV(z)
            r = self.get_R_total(z)

            # 二次方程判定
            delta = v_ocv**2 - 4 * r * p_load

            # 若无解 → 物理降载
            if delta < 0:
                p_load *= 0.5
                delta = v_ocv**2 - 4 * r * p_load
                if delta < 0:
                    p_load = 0.01
                    delta = v_ocv**2 - 4 * r * p_load
                    if delta < 0:
                        break  # 彻底无法供能

            i = (v_ocv - np.sqrt(delta)) / (2 * r)
            v_term = v_ocv - i * r

            if v_term < self.V_cutoff:
                p_load = 0.01
                i = p_load / max(v_term, 2.0)

            # 记录
            res['t_min'].append(t / 60)
            res['soc'].append(max(0, soc))
            res['p_actual'].append(p_load)
            res['v_term'].append(v_term)

            # SOC 更新
            z -= (i * dt) / self.Q_nom
            z = max(z, 0)
            t += dt

        return res

sim = FinalThrottlingSim()
data = sim.run(p_base=2.2)

plt.figure(figsize=(10, 6), dpi=100)
plt.plot(data['t_min'], data['soc'], color='#2c3e50', linewidth=2.5, label='Display SOC')

# 动态背景区域计算 
t_20 = data['t_min'][np.argmin(np.abs(np.array(data['soc'])-20))]
t_10 = data['t_min'][np.argmin(np.abs(np.array(data['soc'])-10))]
t_1 = data['t_min'][np.argmin(np.abs(np.array(data['soc'])-1))]

plt.axvspan(0, t_20, color='green', alpha=0.1, label='Normal Mode')
plt.axvspan(t_20, t_10, color='orange', alpha=0.1, label='Power Saving')
plt.axvspan(t_10, t_1, color='red', alpha=0.1, label='Deep Throttling')
plt.axvspan(t_1, data['t_min'][-1], color='purple', alpha=0.1, label='Survival Mode')

plt.title('Battery SOC Decay ', fontsize=14)
plt.xlabel('Time (Minutes)', fontsize=12)
plt.ylabel('Display SOC (%)', fontsize=12)
plt.ylim(0, 100)
plt.legend(loc='upper right')
plt.grid(True, linestyle='--', alpha=0.5)

plt.annotate('20% Throttling', xy=(t_20, 20), xytext=(+20, +20), 
             textcoords='offset points', arrowprops=dict(arrowstyle='->'))

plt.tight_layout()
plt.show()