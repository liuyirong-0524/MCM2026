import h5py
import matplotlib.pyplot as plt
import numpy as np

path = r'C:\Users\27117\OneDrive\Desktop\archive\2018-04-12_batchdata_updated_struct_errorcorrect.mat'

with h5py.File(path, 'r') as f:
    batch = f['batch']
    cell_index = 0   
    
    # 1. 获取该 Cell 的数据组
    cell_group = f[batch['cycles'][0, cell_index]]
    
    # 2. 定义你想观察的循环次数列表
    # 比如：第 10, 200, 400, 600, 800, 1000 次循环
    target_cycles = [10, 200, 400, 600, 800, 1000]
    
    plt.figure(figsize=(10, 7))
    
    # 使用颜色映射表，让曲线颜色随循环次数平滑变化
    colors = plt.cm.viridis(np.linspace(0, 1, len(target_cycles)))

    for i, cycle_idx in enumerate(target_cycles):
        try:
            
            # 提取 Qdlin (推荐，图像干净)
            qd_lin_ref = cell_group['Qdlin'][cycle_idx - 1, 0]
            Qd_lin = f[qd_lin_ref][()].flatten()
            
            # 生成标准的电压向量 (3.6V -> 2.0V)
            V_lin = np.linspace(3.6, 2.0, len(Qd_lin))
            
            # 绘图
            plt.plot(Qd_lin, V_lin, color=colors[i], label=f'Cycle {cycle_idx}', linewidth=1.5)
            
        except Exception as e:
            print(f"无法提取第 {cycle_idx} 次循环: {e}")

    # 3. 图表装饰
    plt.xlabel('Discharge Capacity (Ah)', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.title(f'Cell {cell_index} - Multi-cycle V-Q Curves (Degradation Visualization)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(title="Cycle Number")
    
    # 设置坐标轴范围（根据 LFP 电池特性微调）
    plt.ylim(2.0, 3.65)
    
    plt.show()