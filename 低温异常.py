import numpy as np
import matplotlib.pyplot as plt

# Plotting style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['axes.unicode_minus'] = False

# 1. Parameter Definitions
V_cutoff = 3.2  # Hardware shutdown threshold (V)
I = 1.0         # Constant discharge current (A)
z = np.linspace(0.01, 1, 500)  # Physical SOC (z)

# Combined+3 OCV Model Parameters
K0, K1, K2, K3, K4 = 3.7, 0.2, -0.01, 0.05, -0.02

def get_ocv(z_val):
    z_clip = np.clip(z_val, 0.001, 0.999)
    return K0 + K1*z_clip + K2/z_clip + K3*np.log(z_clip) + K4*np.log(1-z_clip)

# Internal Resistance Model
R_base = 0.08
T_ref = 298.15  # 25°C Reference
Ea_Rg = 4000 

def get_resistance(T_celsius):
    T_k = T_celsius + 273.15
    return R_base * np.exp(Ea_Rg * (1/T_k - 1/T_ref))

# 2. Execution and Visualization
temperatures = [25, 0, -10]
colors = ['#2ecc71', '#f1c40f', '#e67e22']

plt.figure(figsize=(10, 6), dpi=100)

for T, col in zip(temperatures, colors):
    R = get_resistance(T)
    V_term = get_ocv(z) - I * R
    plt.plot(z * 100, V_term, label=f'Temp: {T}°C ($R \\approx {R:.2f}\\Omega$)', color=col, lw=2.5)

# --- Vertical "Voltage Plunge" Line with Downward Arrow ---
z_target = 0.25  # 25% SOC
v_warm = get_ocv(z_target) - I * get_resistance(25)
v_cold = get_ocv(z_target) - I * get_resistance(-10)

# Draw the dashed vertical line
plt.vlines(x=z_target*100, ymin=v_cold, ymax=v_warm, color='black', linestyle='--', lw=2)

# Add the downward arrow head at the bottom (v_cold)
plt.annotate('', xy=(z_target*100, v_cold), xytext=(z_target*100, v_warm),
             arrowprops=dict(arrowstyle="->", color='black', lw=2, mutation_scale=20))

# Marker points
plt.plot(z_target*100, v_warm, 'ko', markersize=4)
plt.plot(z_target*100, v_cold, 'ko', markersize=4)

# Label for the plunge
plt.annotate('Voltage Plunge\n($\Delta V = I \cdot \Delta R$)', 
             xy=(z_target*100, (v_warm + v_cold)/2), 
             xytext=(z_target*100 - 18, (v_warm + v_cold)/2),
             arrowprops=dict(arrowstyle="->", color='black'),
             color='black', fontweight='bold', ha='center')
# ---------------------------------------------------------

plt.axhline(V_cutoff, color='black', linestyle='--', lw=1.5, label='Shutdown Threshold $V_{cutoff}$')

plt.title('Voltage "Plunge" and Apparent Capacity Loss in Cold Environments', fontsize=14)
plt.xlabel('Physical State of Charge (SOC) [$z$] (%)', fontsize=12)
plt.ylabel('Terminal Voltage ($V_{term}$) [V]', fontsize=12)
plt.ylim(2.4, 4.4)
plt.xlim(0, 100)
plt.legend(loc='lower right', frameon=True, fontsize=10)
plt.grid(True, alpha=0.3)

plt.annotate('Apparent Capacity Loss Zone', xy=(25, 3.15), xytext=(45, 2.7),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=11, color='black', fontweight='bold')

plt.tight_layout()
plt.show()