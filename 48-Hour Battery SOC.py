import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import random

# --- 1. 参数设置 ---
Q_CELL = 3600 * 5.0     # 5000mAh (Coulombs)
R_S = 0.12              # 内阻
ETA_DIS = 0.98          # 放电效率
ETA_CHG = 0.99          # 充电效率
K = [3.7, 0.3, -0.01, 0.04, -0.02] # OCV 系数

P_MODES = {
    'sleep': 0.05, 'background': 0.4, 
    'p1_call': 1.2, 'p2_camera': 2.8, 'p3_navi': 2.2, 
    'p4_social': 1.0, 'p5_video': 1.8, 'p6_gaming': 4.5,
    'charging': -25.0   # 25W 快充
}

# --- 2. 随机负载表 ---
def generate_daily_schedule():
    sleep_slots = ['sleep'] * 8
    activities = ['p1_call', 'p2_camera', 'p3_navi', 'p4_social', 'p5_video', 'p6_gaming']
    active_slots = [random.choice(activities) for _ in range(8)]
    bg_slots = ['background'] * 8
    waking_hours = active_slots + bg_slots
    random.shuffle(waking_hours)
    return sleep_slots + waking_hours

full_schedule = generate_daily_schedule() + generate_daily_schedule()

# --- 3. 动力学模型 ---
def get_vocv(z):
    z = np.clip(z, 0.001, 0.999) 
    return K[0] + K[1]*z + K[2]/z + K[3]*np.log(z) + K[4]*np.log(1-z)

def battery_dynamics(t, y, mode):
    z = y[0]
    if mode == 'charging':
        p_load = P_MODES['charging']
        eff = ETA_CHG
    else:
        hour_idx = int(t / 3600)
        hour_idx = min(hour_idx, 47) 
        p_load = P_MODES[full_schedule[hour_idx]]
        eff = ETA_DIS

    vocv = get_vocv(z)
    discriminant = vocv**2 - 4 * R_S * p_load
    
    if discriminant < 0 or z <= 0.001:
        return [0] 
    
    i_t = (vocv - np.sqrt(discriminant)) / (2 * R_S)
    dz_dt = -(eff * i_t) / Q_CELL
    return [dz_dt]

# --- 4. 事件定义 ---
def event_reach_20(t, y): return y[0] - 0.20
event_reach_20.terminal = True; event_reach_20.direction = -1  

def event_reach_100(t, y): return y[0] - 1.0
event_reach_100.terminal = True; event_reach_100.direction = 1 

def event_empty(t, y): return y[0] - 0.001
event_empty.terminal = True; event_empty.direction = -1

# --- 5. 平滑数据生成辅助函数 ---
# 这是一个新加的函数，专门用来把求解结果变平滑
def get_smooth_segment(sol, points_per_hour=3600): # 默认每秒采样一次，极其平滑
    duration_hours = (sol.t[-1] - sol.t[0]) / 3600
    num_points = int(duration_hours * points_per_hour) + 10 # 保证点数足够
    
    # 创建稠密的时间网格
    t_dense = np.linspace(sol.t[0], sol.t[-1], num_points)
    # 利用 dense_output 插值器计算 SOC
    z_dense = sol.sol(t_dense)[0]
    return t_dense, z_dense

# --- 6. 分段仿真 ---
t_end = 48 * 3600
results_t = []
results_z = []

# === 阶段 1: 初始放电 ===
current_t = 0
current_y = [0.95]

sol1 = solve_ivp(
    lambda t, y: battery_dynamics(t, y, 'discharging'),
    (current_t, t_end), current_y,
    events=event_reach_20,
    max_step=300,        # 放宽步长限制，因为我们有 dense_output 保证平滑
    dense_output=True,   # <--- 关键：启用连续输出
    method='RK45'
)

# 获取平滑数据并存储
t_seg, z_seg = get_smooth_segment(sol1)
results_t.append(t_seg)
results_z.append(z_seg)

current_t = sol1.t[-1]
current_y = [sol1.y[0][-1]]

# === 阶段 2: 充电 ===
if sol1.status == 1 and current_t < t_end:
    sol2 = solve_ivp(
        lambda t, y: battery_dynamics(t, y, 'charging'),
        (current_t, t_end), current_y,
        events=event_reach_100,
        dense_output=True, # <--- 关键
        method='RK45'
    )
    t_seg, z_seg = get_smooth_segment(sol2)
    results_t.append(t_seg)
    results_z.append(z_seg)
    
    current_t = sol2.t[-1]
    current_y = [sol2.y[0][-1]]

    # === 阶段 3: 继续放电 ===
    if sol2.status == 1 and current_t < t_end:
        sol3 = solve_ivp(
            lambda t, y: battery_dynamics(t, y, 'discharging'),
            (current_t, t_end), current_y,
            events=event_empty,
            dense_output=True, # <--- 关键
            method='RK45'
        )
        t_seg, z_seg = get_smooth_segment(sol3)
        results_t.append(t_seg)
        results_z.append(z_seg)

# 合并数据
t_total = np.concatenate(results_t)
z_total = np.concatenate(results_z)

# --- 7. 可视化 (增加美观度) ---
plt.figure(figsize=(12, 6), dpi=100) # 增加DPI使图像更清晰

# 绘制 SOC 曲线
plt.plot(t_total / 3600, z_total * 100, color='#1f77b4', linewidth=2, label='SOC ($z$)')

# 填充充电区域的背景色，使其看起来更直观
# 我们通过判断导数(或SOC是否上升)来寻找充电区间


# 辅助线
plt.axhline(y=20, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Trigger (20%)')
plt.axhline(y=100, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Target (100%)')

plt.title('48-Hour Battery SOC: Smooth High-Res Simulation', fontsize=14, pad=15)
plt.xlabel('Time (Hours)', fontsize=12)
plt.ylabel('State of Charge (%)', fontsize=12)
plt.xlim(0, 48)
plt.ylim(0, 105)
plt.xticks(np.arange(0, 49, 4))
plt.grid(True, linestyle='-', alpha=0.3) # 网格线淡一点
plt.legend(loc='upper right', frameon=True, shadow=True)

plt.tight_layout()
plt.show()