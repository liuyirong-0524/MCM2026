import numpy as np
import matplotlib.pyplot as plt
import random

# --- 1. 电池模型参数 ---
Q_CELL = 3600 * 5.0     # 5000mAh (Coulombs)
R_S = 0.12              # 内阻 (Ohms)
ETA_DIS = 0.98
ETA_CHG = 0.99
K = [3.7, 0.3, -0.01, 0.04, -0.02] # OCV 系数
V_LIMIT = 2.5           # 实际关机保护触发线（机制保持不变）

# 功率模式
P_MODES = {
    'sleep': 0.05, 'background': 0.4, 
    'p1_call': 1.2, 'p2_camera': 2.8, 'p3_navi': 2.2, 
    'p4_social': 1.0, 'p5_video': 1.8, 'p6_gaming': 4.5,
    'charging': -25.0
}

# --- 2. 辅助函数：OCV 及其导数 ---
def get_vocv(z):
    z = np.clip(z, 0.001, 0.999)
    return K[0] + K[1]*z + K[2]/z + K[3]*np.log(z) + K[4]*np.log(1-z)

def get_dvocv_dz(z):
    z = np.clip(z, 0.001, 0.999)
    return K[1] - K[2]/(z**2) + K[3]/z - K[4]/(1-z)

# --- 3. 生成真实数据 (Ground Truth) ---
def generate_ground_truth():
    def get_schedule():
        slots = ['sleep']*16 + ['background']*16 + [random.choice(list(P_MODES.keys())[:-1]) for _ in range(16)]
        random.shuffle(slots)
        return slots + slots # 48h
    schedule = get_schedule()

    dt = 1.0
    steps = int(48 * 3600 / dt)
    
    I_true = np.zeros(steps)
    z_true = np.zeros(steps)
    v_true = np.zeros(steps) 
    
    z = 0.95
    has_charged = False
    is_charging = False
    is_shutdown = False
    
    for i in range(steps):
        t = i * dt
        if z < 0.20 and not has_charged: 
            is_charging = True
            has_charged = True
            is_shutdown = False
        if is_charging and z >= 1.0: 
            is_charging = False
            z = 1.0
        
        if is_charging:
            p_load = P_MODES['charging']; eff = ETA_CHG
        elif is_shutdown:
            p_load = 0.0; eff = 1.0
        else:
            p_load = P_MODES[schedule[min(int(t/1800), 95)]]; eff = ETA_DIS
            
        vocv = get_vocv(z)
        disc = vocv**2 - 4 * R_S * p_load
        i_curr = 0 if disc < 0 else (vocv - np.sqrt(disc)) / (2 * R_S)
        v_term = vocv - i_curr * R_S
        
        # 维持 2.5V 保护机制不变
        if not is_charging and v_term < V_LIMIT:
            is_shutdown = True
            i_curr = 0.0
            v_term = vocv
            
        z = z - (eff * i_curr / Q_CELL) * dt
        z = np.clip(z, 0.001, 1.0)
        
        I_true[i] = i_curr
        z_true[i] = z
        v_true[i] = v_term
        
    return I_true, z_true, v_true

# --- 4. 扩展卡尔曼滤波 (EKF) 实现 ---
def run_ekf(I_meas, V_meas, dt):
    steps = len(I_meas)
    x_est = 0.80  # 初始猜 0.80
    P_est = 0.1   
    Q_cov = 1e-7  
    R_cov = 1e-2  
    ekf_soc = np.zeros(steps)
    for k in range(steps):
        i_k = I_meas[k]
        eff = ETA_CHG if i_k < 0 else ETA_DIS
        x_pred = np.clip(x_est - (eff * i_k * dt) / Q_CELL, 0.001, 0.999)
        P_pred = P_est + Q_cov
        v_model_pred = get_vocv(x_pred) - i_k * R_S
        y_residual = V_meas[k] - v_model_pred
        H = get_dvocv_dz(x_pred)
        S = H * P_pred * H + R_cov
        K_gain = P_pred * H / S
        x_est = np.clip(x_pred + K_gain * y_residual, 0.001, 0.999)
        P_est = (1 - K_gain * H) * P_pred
        ekf_soc[k] = x_est
    return ekf_soc

# --- 5. 执行 ---
I_true, z_true, v_true = generate_ground_truth()
v_measured = v_true + np.random.normal(0, 0.015, len(v_true))
ekf_soc = run_ekf(I_true, v_measured, dt=1.0)

# --- 6. 可视化结果 (增加 3.4V 关机线标识) ---
t_hours = np.linspace(0, 48, len(z_true))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 图1: 电流输入
ax1.plot(t_hours, I_true, color='orange', label='Current $I(t)$')
ax1.set_ylabel('Current (A)')
ax1.set_title('Step 1: Input Current $I(t)$', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(loc='upper right')

# 图2: 电压 (添加 3.4V 关机线)
ax2.plot(t_hours, v_measured, color='gray', alpha=0.5, label='Measured Voltage (Noisy)')
ax2.plot(t_hours, v_true, color='green', linewidth=1, label='True Voltage')

# 仅在此处增加 3.4V 的关机线标识
ax2.axhline(3.4, color='blue', linestyle='--', linewidth=1.2, label='Shutdown Line (3.4V)')
ax2.axhline(2.5, color='red', linestyle=':', linewidth=1.2, label='Danger Line (2.5V)')

ax2.set_ylabel('Voltage (V)')
ax2.set_title('Step 2: Voltage Measurement (Observation)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper right')

# 图3: SOC (保持原有标识和标注)
ax3.plot(t_hours, z_true * 100, 'k--', linewidth=2, label='True SOC (Reference)')
ax3.plot(t_hours, ekf_soc * 100, 'r-', linewidth=1.5, label='EKF Estimated SOC')
ax3.set_ylabel('SOC (%)')
ax3.set_xlabel('Time (Hours)')
ax3.set_title('Step 3: EKF Estimation Results', fontsize=12)
ax3.grid(True, alpha=0.3)
ax3.legend(loc='best')
ax3.set_ylim(0, 105)

# 标注初始收敛 (保持一致)
ax3.annotate('Initial Correction\n(Guess: 80% -> True: 95%)', 
             xy=(0.5, 85), xytext=(3, 60),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()
plt.show()