import numpy as np
import matplotlib.pyplot as plt
import random

# --- 1. 电池模型参数 ---
Q_CELL = 3600 * 5.0     # 5000mAh (Coulombs)
R_S = 0.12              # 内阻 (Ohms)
ETA_DIS = 0.98
ETA_CHG = 0.99
K = [3.7, 0.3, -0.01, 0.04, -0.02] # OCV 系数
V_LIMIT = 2.5           # 【新增】电压保护阈值 (Volts)

# 功率模式
P_MODES = {
    'sleep': 0.05, 'background': 0.4, 
    'p1_call': 1.2, 'p2_camera': 2.8, 'p3_navi': 2.2, 
    'p4_social': 1.0, 'p5_video': 1.8, 'p6_gaming': 4.5,
    'charging': -25.0
}

# --- 2. 辅助函数 ---
def get_vocv(z):
    z = np.clip(z, 0.001, 0.999)
    return K[0] + K[1]*z + K[2]/z + K[3]*np.log(z) + K[4]*np.log(1-z)

def get_dvocv_dz(z):
    z = np.clip(z, 0.001, 0.999)
    return K[1] - K[2]/(z**2) + K[3]/z - K[4]/(1-z)

# --- 3. 生成真实数据 (加入电压保护逻辑) ---
def generate_ground_truth():
    def get_schedule():
        slots = ['sleep']*16 + ['background']*16 + [random.choice(list(P_MODES.keys())[:-1]) for _ in range(16)]
        random.shuffle(slots)
        return slots + slots 
    schedule = get_schedule()

    dt = 1.0
    steps = int(48 * 3600 / dt)
    
    I_true = np.zeros(steps)
    z_true = np.zeros(steps)
    v_true = np.zeros(steps)
    
    z = 0.95
    is_charging = False
    is_shutdown = False # 【新增】关机状态标记
    
    for i in range(steps):
        t = i * dt
        
        # 1. 充电逻辑控制
        # 如果 SOC 极低或已关机，触发充电逻辑（模拟用户插枪）
        if (z < 0.05 or is_shutdown) and not is_charging:
            # 只有在极低电量或关机后一段时间模拟插电
            if random.random() < 0.01: # 模拟用户随机插上充电器
                is_charging = True
                is_shutdown = False 
        
        if is_charging and z >= 1.0:
            is_charging = False
            z = 1.0
        
        # 2. 确定当前功率需求
        if is_charging:
            p_load = P_MODES['charging']; eff = ETA_CHG
        elif is_shutdown:
            p_load = 0.0; eff = 1.0 # 关机状态，无功耗
        else:
            p_load = P_MODES[schedule[min(int(t/1800), 95)]]; eff = ETA_DIS
            
        # 3. 计算电流与端电压
        vocv = get_vocv(z)
        disc = vocv**2 - 4 * R_S * p_load
        
        if disc < 0: # 电池无法提供所需功率
            i_curr = vocv / (2 * R_S) # 极限电流
        else:
            i_curr = (vocv - np.sqrt(disc)) / (2 * R_S)
        
        v_term = vocv - i_curr * R_S
        
        # --- 【核心改动：电压阈值检测】 ---
        if v_term < V_LIMIT and not is_charging:
            is_shutdown = True
            i_curr = 0.0
            v_term = get_vocv(z) # 关机后电流消失，电压回升到开路电压
            # print(f"Warning: Battery reached {V_LIMIT}V at step {i}. Emergency Shutdown!")

        # 4. 更新状态
        z = z - (eff * i_curr / Q_CELL) * dt
        z = np.clip(z, 0.0001, 1.0)
        
        I_true[i] = i_curr
        z_true[i] = z
        v_true[i] = v_term
        
    return I_true, z_true, v_true

# --- 4. EKF 实现 (保持不变) ---
def run_ekf(I_meas, V_meas, dt):
    steps = len(I_meas)
    x_est = 0.80 
    P_est = 0.1  
    Q_cov = 1e-7 
    R_cov = 1e-2 
    ekf_soc = np.zeros(steps)
    
    for k in range(steps):
        i_k = I_meas[k]
        eff = ETA_CHG if i_k < 0 else ETA_DIS
        x_pred = x_est - (eff * i_k * dt) / Q_CELL
        x_pred = np.clip(x_pred, 0.001, 0.999)
        P_pred = P_est + Q_cov
        
        v_measured_k = V_meas[k]
        v_model_pred = get_vocv(x_pred) - i_k * R_S
        y_residual = v_measured_k - v_model_pred
        H = get_dvocv_dz(x_pred)
        S = H * P_pred * H + R_cov
        K_gain = P_pred * H / S
        
        x_est = x_pred + K_gain * y_residual
        x_est = np.clip(x_est, 0.001, 0.999)
        P_est = (1 - K_gain * H) * P_pred
        ekf_soc[k] = x_est
        
    return ekf_soc

# --- 5. 执行与可视化 ---
I_true, z_true, v_true = generate_ground_truth()
noise_std = 0.01
v_measured = v_true + np.random.normal(0, noise_std, len(v_true))
ekf_soc = run_ekf(I_true, v_measured, dt=1.0)

t_hours = np.linspace(0, 48, len(z_true))
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# 图1: 电流
ax1.plot(t_hours, I_true, color='orange')
ax1.set_ylabel('Current (A)')
ax1.set_title('Step 1: Input Current $I(t)$ (Shutdown when V < 2.5V)')
ax1.grid(True, alpha=0.3)

# 图2: 电压 (添加阈值线)
ax2.plot(t_hours, v_true, color='green', label='True Voltage')
ax2.axhline(y=V_LIMIT, color='red', linestyle='--', label='Damage Threshold (2.5V)')
ax2.set_ylabel('Voltage (V)')
ax2.set_title('Step 2: Voltage Measurement & Protection')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

# 图3: SOC
ax3.plot(t_hours, z_true * 100, 'k--', label='True SOC')
ax3.plot(t_hours, ekf_soc * 100, 'r-', alpha=0.7, label='EKF SOC')
ax3.set_ylabel('SOC (%)')
ax3.set_xlabel('Time (Hours)')
ax3.set_title('Step 3: SOC Estimation')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()