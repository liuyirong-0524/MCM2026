import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

path = r'C:\Users\27117\OneDrive\Desktop\archive\2018-04-12_batchdata_updated_struct_errorcorrect.mat'

def analyze_cells(cell_indices):
    with h5py.File(path, 'r') as f:
        batch = f['batch']
        
        total_cells = batch['summary'].shape[1]
        valid_indices = [i for i in cell_indices if i < total_cells]
        
        if not valid_indices:
            print(f"Indices {cell_indices} are all out of range. Max index is {total_cells-1}")
            return

        fig, axes = plt.subplots(len(valid_indices), 1, figsize=(12, 5 * len(valid_indices)))
        if len(valid_indices) == 1: axes = [axes]
        
        for idx, cell_idx in enumerate(valid_indices):
            try:
                summary_ref = batch['summary'][0, cell_idx]
                summary_group = f[summary_ref]
                
                keys = list(summary_group.keys())
                
                qd_key = next((k for k in ['QDischarge', 'Qd', 'QD'] if k in keys), None)
                ir_key = next((k for k in ['IR', 'ir'] if k in keys), None)
                cyc_key = 'cycle' if 'cycle' in keys else None

                if not qd_key or not ir_key:
                    print(f"Cell {cell_idx} 缺少关键键。现有键: {keys}")
                    continue

                cycles = summary_group[cyc_key][()].flatten()
                ir = summary_group[ir_key][()].flatten()
                qd = summary_group[qd_key][()].flatten()
                
                # 过滤无效点
                mask = (ir > 0) & (qd > 0.1)
                c_clean, ir_clean, qd_clean = cycles[mask], ir[mask], qd[mask]

                # 平滑处理
                w = 15 if len(c_clean) > 15 else (len(c_clean)//2*2-1 if len(c_clean)>3 else 0)
                ir_s = savgol_filter(ir_clean, w, 2) if w > 3 else ir_clean
                qd_s = savgol_filter(qd_clean, w, 2) if w > 3 else qd_clean

                # 绘图
                ax_ir = axes[idx]
                ax_qd = ax_ir.twinx()
                
                ax_ir.plot(c_clean, ir_s, color='darkorange', lw=2, label='IR')
                ax_ir.set_ylabel(r'Internal Resistance ($\Omega$)', color='darkorange')
                
                ax_qd.plot(c_clean, qd_s, color='royalblue', lw=2, label='Capacity')
                ax_qd.set_ylabel('Discharge Capacity (Ah)', color='royalblue')
                
                ax_ir.set_title(f'Cell {cell_idx} Analysis')
                ax_ir.grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"处理 Cell {cell_idx} 时出错: {e}")

        plt.tight_layout()
        plt.show()

# 调用
analyze_cells([0]) 