import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
class SolverEKFSim:
    def __init__(self):
        self.Q_nom = 4500 * 3600 / 1000 
        self.V_cutoff = 2.5
        self.R_s = 0.15  
        self.z_full = 1.0               
        self.K = [3.8, 0.15, -0.01, 0.08, -0.01] 
        self.k1, self.k2 = 0.05, 10.0
        
        # --- EKF 协方差参数 ---
        self.Q_ekf = 1e-7  # 过程噪声（相信预测的程度）
        self.R_ekf = 1e-3  # 测量噪声（相信电压表的程度）

    def get_R_total(self, z):
        return self.R_s * (1 + 0.05 * np.exp(-10.0 * z))

    def get_OCV(self, z):
        z = np.clip(z, 0.0001, 0.9999)
        return self.K[0] + self.K[1]*z + self.K[2]/z + \
               self.K[3]*np.log(z) + self.K[4]*np.log(1-z)

    def get_beta(self, s):
        if s > 20: return 1
        if s > 10: return 0.8
        if s > 1:  return 0.5
        return 0.1

    def dz_dt(self, z, p_base):
        s = (z / self.z_full) * 100
        p_load = p_base * self.get_beta(s)
        v_ocv = self.get_OCV(z)
        r = self.get_R_total(z)
        delta = v_ocv**2 - 4 * r * p_load
        if delta < 0: return 0
        i = (v_ocv - np.sqrt(delta)) / (2 * r)
        return -i / self.Q_nom

    def ekf_update(self, z_pred, P_prev, p_base):
        v_ocv = self.get_OCV(z_pred)
        v_meas = v_ocv - 0.02 + np.random.normal(0, 0.01) 
        
        z_eps = np.clip(z_pred, 0.01, 0.99)
        H = self.K[1] - self.K[2]/(z_eps**2) + self.K[3]/z_eps - self.K[4]/(1-z_eps)
        
        P_pred = P_prev + self.Q_ekf
        K_gain = P_pred * H / (H * P_pred * H + self.R_ekf)
        
        z_corr = z_pred + K_gain * (v_meas - v_ocv)
        P_corr = (1 - K_gain * H) * P_pred
        
        return np.clip(z_corr, 0, 1), P_corr

    def run_sim(self, p_base, dt, method='euler', use_ekf=False):
        z, t = self.z_full, 0
        P_ekf = 1e-4 # 初始不确定度
        res = {'t': [], 'soc': []}
        
        while z > 0.001:
            res['t'].append(t/60); res['soc'].append(z*100)
            
            # 预测步 (Prediction)
            if method == 'euler':
                slope = self.dz_dt(z, p_base)
                z_next = z + slope * dt
            else: # RK4
                k1 = self.dz_dt(z, p_base)
                k2 = self.dz_dt(z + dt/2 * k1, p_base)
                k3 = self.dz_dt(z + dt/2 * k2, p_base)
                k4 = self.dz_dt(z + dt * k3, p_base)
                z_next = z + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            
            if self.dz_dt(z, p_base) == 0: break
                
            # 修正步 (Correction via EKF)
            if use_ekf:
                z, P_ekf = self.ekf_update(z_next, P_ekf, p_base)
            else:
                z = z_next
                
            t += dt
        return res

# --- 运行对比 ---
sim = SolverEKFSim()
p_test = 1.61
dt_val = 20 

data_euler = sim.run_sim(p_test, dt_val, 'euler', False)
data_rk4 = sim.run_sim(p_test, dt_val, 'rk4', False)
data_euler_ekf = sim.run_sim(p_test, dt_val, 'euler', True)
data_rk4_ekf = sim.run_sim(p_test, dt_val, 'rk4', True)

plt.figure(figsize=(10, 6), dpi=100)

results = [
    (data_euler, 'Euler', 'red', '--'),
    (data_rk4, 'RK4', 'blue', '--'),
    (data_euler_ekf, 'Euler+EKF', 'green', '-'),
    (data_rk4_ekf, 'RK4+EKF', 'magenta', '-')
]

for d, name, col, lstyle in results:
    plt.plot(d['t'], d['soc'], color=col, linestyle=lstyle, label=name)

offsets = [12, -12, 24, -24]  
h_aligns = ['left', 'right', 'center', 'center']

for i, (d, name, col, _) in enumerate(results):
    t_z = d['t'][-1]
    s_z = d['soc'][-1]
    
    plt.scatter(t_z, s_z, color=col, s=40, zorder=5)
    
    plt.annotate(f'{name}\n{t_z:.1f} min', 
                 xy=(t_z, s_z), 
                 xytext=(15 if i%2==0 else -15, offsets[i]), # 交错左右和上下
                 textcoords='offset points',
                 color=col,
                 fontweight='bold',
                 fontsize=9,
                 ha=h_aligns[i],
                 arrowprops=dict(arrowstyle='->', color=col, alpha=0.3))

plt.title('Solver & EKF Comparison', fontsize=12)
plt.xlabel('Time (Minutes)')
plt.ylabel('SOC (%)')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')
plt.ylim(-10, 105) 

plt.tight_layout()
plt.show()