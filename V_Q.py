import h5py
import matplotlib.pyplot as plt
import numpy as np

path = r'C:\Users\27117\OneDrive\Desktop\archive\2018-04-12_batchdata_updated_struct_errorcorrect.mat'

with h5py.File(path, 'r') as f:
    batch = f['batch']
    
    cell_index = 0   
    cycle_index = 10 
    
    # 获取 Cell 组
    cell_ref = batch['cycles'][0, cell_index]
    cell_group = f[cell_ref]
    
    v_dataset = cell_group['V']
    shape = v_dataset.shape
    print(f"Cell {cell_index} 的 'V' 数据集形状为: {shape}")
    
    # 如果形状是 (1, 1) 或 (0, 0)，说明这个 cell 没数据或者索引不对
    if shape == (1, 1) or shape[0] == 0:
        print("警告: 当前 Cell 似乎不包含多个循环数据，正在尝试检查 Cell 1...")
        # 自动切换到第 2 个电池作为演示
        cell_ref = batch['cycles'][0, 1] 
        cell_group = f[cell_ref]
        v_dataset = cell_group['V']
        print(f"切换到 Cell 1，'V' 形状为: {v_dataset.shape}")

    # --- 动态计算索引 ---
    # 判定是 (1, N) 还是 (N, 1) 结构
    rows, cols = v_dataset.shape
    max_cycles = max(rows, cols)
    
    if cycle_index > max_cycles:
        print(f"错误: 该电池仅有 {max_cycles} 次循环，无法访问第 {cycle_index} 次。")
        cycle_to_use = max_cycles
    else:
        cycle_to_use = cycle_index

    print(f"正在提取第 {cycle_to_use} 次循环的数据...")

    # 根据形状自动调整索引位置
    if rows < cols: # (1, N) 结构
        v_ref = v_dataset[0, cycle_to_use - 1]
        qd_ref = cell_group['Qd'][0, cycle_to_use - 1]
    else: # (N, 1) 结构
        v_ref = v_dataset[cycle_to_use - 1, 0]
        qd_ref = cell_group['Qd'][cycle_to_use - 1, 0]

    # --- 解引用并绘图 ---
    V = f[v_ref][()].flatten()
    Qd = f[qd_ref][()].flatten()

    plt.figure(figsize=(8, 5))
    plt.plot(Qd, V, label=f'Cycle {cycle_to_use}')
    plt.xlabel('Discharge Capacity (Ah)')
    plt.ylabel('Voltage (V)')
    plt.title(f'Cell {cell_index} - V-Q Curve (Fixed Indexing)')
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.show()