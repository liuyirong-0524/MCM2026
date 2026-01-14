"""
生态系统3R指标空间化可视化程序
包含空间网格离散化、扩散效应、空间异质性等特性
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class SpatialEcosystemModel:
    """空间化生态系统3R模型"""

    def __init__(self, grid_size=50, resistance_map=None, resilience_map=None,
                 recovery_rate_map=None, diffusion_rate=0.1):
        """
        参数：
        - grid_size: 网格大小 (grid_size × grid_size)
        - resistance_map: 抵抗力空间分布 (grid_size × grid_size)，若为None则使用均匀值
        - resilience_map: 恢复力空间分布
        - recovery_rate_map: 恢复速度空间分布
        - diffusion_rate: 空间扩散系数 (0-1)，控制邻近网格的相互影响
        """
        self.grid_size = grid_size
        self.diffusion_rate = diffusion_rate
        self.initial_state = 100.0

        # 初始化3R参数的空间分布
        if resistance_map is None:
            self.resistance_map = np.ones((grid_size, grid_size)) * 0.7
        else:
            self.resistance_map = resistance_map

        if resilience_map is None:
            self.resilience_map = np.ones((grid_size, grid_size)) * 0.8
        else:
            self.resilience_map = resilience_map

        if recovery_rate_map is None:
            self.recovery_rate_map = np.ones((grid_size, grid_size)) * 0.3
        else:
            self.recovery_rate_map = recovery_rate_map

        # 初始化状态网格
        self.state = np.ones((grid_size, grid_size)) * self.initial_state
        self.history = []  # 存储历史状态

    def create_disturbance_mask(self, disturbance_type='circular',
                                center=None, radius=10, intensity=50):
        """
        创建干扰掩码

        参数：
        - disturbance_type: 'circular', 'rectangular', 'random', 'gradient'
        - center: 干扰中心位置 (x, y)
        - radius: 干扰半径（对圆形和矩形）
        - intensity: 干扰强度 (0-100)
        """
        mask = np.zeros((self.grid_size, self.grid_size))

        if center is None:
            center = (self.grid_size // 2, self.grid_size // 2)

        if disturbance_type == 'circular':
            # 圆形干扰
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    if dist <= radius:
                        # 距离中心越近，干扰越强
                        mask[i, j] = intensity * (1 - dist / radius)

        elif disturbance_type == 'rectangular':
            # 矩形干扰
            x1, y1 = max(0, center[0] - radius), max(0, center[1] - radius)
            x2, y2 = min(self.grid_size, center[0] + radius), min(self.grid_size, center[1] + radius)
            mask[x1:x2, y1:y2] = intensity

        elif disturbance_type == 'random':
            # 随机斑块干扰
            n_patches = 5
            for _ in range(n_patches):
                cx = np.random.randint(0, self.grid_size)
                cy = np.random.randint(0, self.grid_size)
                r = np.random.randint(5, 15)
                for i in range(self.grid_size):
                    for j in range(self.grid_size):
                        dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                        if dist <= r:
                            mask[i, j] = max(mask[i, j], intensity * np.random.uniform(0.5, 1.0))

        elif disturbance_type == 'gradient':
            # 梯度干扰（从左到右递减）
            for j in range(self.grid_size):
                mask[:, j] = intensity * (1 - j / self.grid_size)

        return mask

    def apply_diffusion(self, state):
        """
        应用空间扩散效应（邻近网格相互影响）
        使用简单的拉普拉斯算子
        """
        if self.diffusion_rate == 0:
            return state

        new_state = state.copy()

        # 使用卷积实现扩散
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                # 四邻域平均
                neighbors_avg = (state[i-1, j] + state[i+1, j] +
                                state[i, j-1] + state[i, j+1]) / 4.0
                # 扩散：当前值向邻域平均靠拢
                new_state[i, j] = state[i, j] + self.diffusion_rate * (neighbors_avg - state[i, j])

        return new_state

    def simulate_step(self, disturbance_mask=None, dt=0.1):
        """
        模拟一个时间步

        参数：
        - disturbance_mask: 干扰强度的空间分布
        - dt: 时间步长
        """
        # 应用干扰
        if disturbance_mask is not None:
            decline = disturbance_mask * (1 - self.resistance_map)
            self.state = np.maximum(self.state - decline, 0)

        # 恢复过程
        target_state = self.initial_state * self.resilience_map
        self.state = self.state + (target_state - self.state) * self.recovery_rate_map * dt

        # 应用空间扩散
        self.state = self.apply_diffusion(self.state)

        # 记录历史
        self.history.append(self.state.copy())

        return self.state

    def simulate(self, total_time=100, dt=0.1, disturbance_time=20,
                 disturbance_type='circular', disturbance_intensity=50):
        """
        完整模拟过程

        返回：
        - history: 状态历史列表
        """
        self.history = []
        self.state = np.ones((self.grid_size, self.grid_size)) * self.initial_state

        time_steps = int(total_time / dt)

        for step in range(time_steps):
            t = step * dt

            # 在特定时间施加干扰
            if abs(t - disturbance_time) < dt:
                disturbance_mask = self.create_disturbance_mask(
                    disturbance_type=disturbance_type,
                    intensity=disturbance_intensity
                )
            else:
                disturbance_mask = None

            self.simulate_step(disturbance_mask, dt)

        return self.history


class SpatialVisualization:
    """空间化3R指标交互式可视化"""

    def __init__(self, grid_size=50):
        self.grid_size = grid_size

        # 创建图形界面
        self.fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, figure=self.fig, hspace=0.35, wspace=0.3)

        # 主图：当前状态热力图
        self.ax_current = self.fig.add_subplot(gs[0:2, 0:2])

        # 3R参数分布
        self.ax_resistance = self.fig.add_subplot(gs[0, 2])
        self.ax_resilience = self.fig.add_subplot(gs[1, 2])

        # 统计图
        self.ax_stats = self.fig.add_subplot(gs[2, :2])
        self.ax_hist = self.fig.add_subplot(gs[2, 2])

        # 初始化参数
        self.base_resistance = 0.7
        self.base_resilience = 0.8
        self.base_recovery = 0.3
        self.diffusion_rate = 0.1
        self.disturbance_intensity = 50
        self.heterogeneity = 0.2  # 空间异质性程度

        # 干扰类型
        self.disturbance_types = ['circular', 'rectangular', 'random', 'gradient']
        self.current_disturbance_idx = 0

        # 创建模型
        self.model = None
        self.history = None
        self.current_time_step = 0

        # 创建控制界面
        self.create_controls()

        # 初始化模拟
        self.run_simulation()

    def create_heterogeneous_map(self, base_value, heterogeneity):
        """
        创建具有空间异质性的参数分布图

        参数：
        - base_value: 基础值
        - heterogeneity: 异质性程度 (0-1)
        """
        # 使用多个正弦波叠加创建平滑的空间变化
        x = np.linspace(0, 4 * np.pi, self.grid_size)
        y = np.linspace(0, 4 * np.pi, self.grid_size)
        X, Y = np.meshgrid(x, y)

        # 叠加多个不同频率的波
        pattern = (np.sin(X) * np.cos(Y) +
                  0.5 * np.sin(2 * X + 1) * np.cos(2 * Y + 1) +
                  0.3 * np.sin(3 * X + 2) * np.cos(3 * Y + 2))

        # 归一化到 [-1, 1]
        pattern = pattern / np.max(np.abs(pattern))

        # 转换到目标范围
        result = base_value + heterogeneity * pattern * base_value
        result = np.clip(result, 0, 1)

        return result

    def create_controls(self):
        """创建控制滑块和按钮"""
        slider_color = 'lightgoldenrodyellow'

        # 基础3R参数
        ax_res = plt.axes([0.15, 0.24, 0.25, 0.015], facecolor=slider_color)
        self.slider_resistance = Slider(ax_res, 'Resistance', 0.0, 1.0,
                                       valinit=self.base_resistance, valstep=0.05)

        ax_resil = plt.axes([0.15, 0.22, 0.25, 0.015], facecolor=slider_color)
        self.slider_resilience = Slider(ax_resil, 'Resilience', 0.0, 1.0,
                                       valinit=self.base_resilience, valstep=0.05)

        ax_recov = plt.axes([0.15, 0.20, 0.25, 0.015], facecolor=slider_color)
        self.slider_recovery = Slider(ax_recov, 'Recovery', 0.0, 1.0,
                                     valinit=self.base_recovery, valstep=0.05)

        # 空间参数
        ax_diff = plt.axes([0.15, 0.18, 0.25, 0.015], facecolor=slider_color)
        self.slider_diffusion = Slider(ax_diff, 'Diffusion', 0.0, 0.5,
                                      valinit=self.diffusion_rate, valstep=0.05)

        ax_hetero = plt.axes([0.15, 0.16, 0.25, 0.015], facecolor=slider_color)
        self.slider_heterogeneity = Slider(ax_hetero, 'Heterogeneity', 0.0, 0.5,
                                          valinit=self.heterogeneity, valstep=0.05)

        # 干扰参数
        ax_intens = plt.axes([0.15, 0.14, 0.25, 0.015], facecolor=slider_color)
        self.slider_intensity = Slider(ax_intens, 'Disturbance', 0, 100,
                                      valinit=self.disturbance_intensity, valstep=5)

        # 时间滑块
        ax_time = plt.axes([0.15, 0.10, 0.25, 0.015], facecolor=slider_color)
        self.slider_time = Slider(ax_time, 'Time Step', 0, 99,
                                 valinit=0, valstep=1)

        # 按钮
        ax_button_run = plt.axes([0.15, 0.06, 0.08, 0.03])
        self.button_run = Button(ax_button_run, 'Run Simulation')

        ax_button_disturb = plt.axes([0.25, 0.06, 0.15, 0.03])
        self.button_disturb = Button(ax_button_disturb, 'Disturbance: Circular')

        # 绑定事件
        self.slider_resistance.on_changed(self.on_param_change)
        self.slider_resilience.on_changed(self.on_param_change)
        self.slider_recovery.on_changed(self.on_param_change)
        self.slider_diffusion.on_changed(self.on_param_change)
        self.slider_heterogeneity.on_changed(self.on_param_change)
        self.slider_intensity.on_changed(self.on_param_change)
        self.slider_time.on_changed(self.on_time_change)
        self.button_run.on_clicked(self.run_simulation)
        self.button_disturb.on_clicked(self.toggle_disturbance_type)

    def toggle_disturbance_type(self, event):
        """切换干扰类型"""
        self.current_disturbance_idx = (self.current_disturbance_idx + 1) % len(self.disturbance_types)
        disturbance_type = self.disturbance_types[self.current_disturbance_idx]
        self.button_disturb.label.set_text(f'Disturbance: {disturbance_type.capitalize()}')
        self.run_simulation(event)

    def on_param_change(self, val):
        """参数改变时的回调"""
        self.base_resistance = self.slider_resistance.val
        self.base_resilience = self.slider_resilience.val
        self.base_recovery = self.slider_recovery.val
        self.diffusion_rate = self.slider_diffusion.val
        self.heterogeneity = self.slider_heterogeneity.val
        self.disturbance_intensity = self.slider_intensity.val

    def on_time_change(self, val):
        """时间滑块改变时更新显示"""
        if self.history is not None:
            self.current_time_step = int(self.slider_time.val)
            self.update_display()

    def run_simulation(self, event=None):
        """运行模拟"""
        # 创建空间异质性的3R参数分布
        resistance_map = self.create_heterogeneous_map(self.base_resistance, self.heterogeneity)
        resilience_map = self.create_heterogeneous_map(self.base_resilience, self.heterogeneity)
        recovery_map = self.create_heterogeneous_map(self.base_recovery, self.heterogeneity)

        # 创建模型
        self.model = SpatialEcosystemModel(
            grid_size=self.grid_size,
            resistance_map=resistance_map,
            resilience_map=resilience_map,
            recovery_rate_map=recovery_map,
            diffusion_rate=self.diffusion_rate
        )

        # 运行模拟
        disturbance_type = self.disturbance_types[self.current_disturbance_idx]
        self.history = self.model.simulate(
            total_time=100,
            dt=1.0,
            disturbance_time=20,
            disturbance_type=disturbance_type,
            disturbance_intensity=self.disturbance_intensity
        )

        # 更新时间滑块范围
        self.slider_time.valmax = len(self.history) - 1
        self.slider_time.ax.set_xlim(0, len(self.history) - 1)

        # 重置时间步
        self.current_time_step = 0
        self.slider_time.set_val(0)

        # 更新显示
        self.update_display()

    def update_display(self):
        """更新所有显示"""
        if self.history is None:
            return

        current_state = self.history[self.current_time_step]

        # 更新当前状态热力图
        self.ax_current.clear()
        im = self.ax_current.imshow(current_state, cmap='RdYlGn', vmin=0, vmax=100,
                                    origin='lower', interpolation='bilinear')
        self.ax_current.set_title(f'Ecosystem State at t={self.current_time_step}',
                                 fontsize=12, fontweight='bold')
        self.ax_current.set_xlabel('X coordinate')
        self.ax_current.set_ylabel('Y coordinate')

        # 添加颜色条
        if not hasattr(self, 'colorbar'):
            self.colorbar = plt.colorbar(im, ax=self.ax_current)
            self.colorbar.set_label('State Value', rotation=270, labelpad=15)

        # 更新抵抗力分布
        self.ax_resistance.clear()
        self.ax_resistance.imshow(self.model.resistance_map, cmap='Blues', vmin=0, vmax=1,
                                 origin='lower')
        self.ax_resistance.set_title('Resistance Map', fontsize=10)
        self.ax_resistance.axis('off')

        # 更新恢复力分布
        self.ax_resilience.clear()
        self.ax_resilience.imshow(self.model.resilience_map, cmap='Greens', vmin=0, vmax=1,
                                 origin='lower')
        self.ax_resilience.set_title('Resilience Map', fontsize=10)
        self.ax_resilience.axis('off')

        # 更新统计图：平均状态随时间变化
        self.ax_stats.clear()
        mean_states = [np.mean(state) for state in self.history]
        std_states = [np.std(state) for state in self.history]
        time_points = np.arange(len(self.history))

        self.ax_stats.plot(time_points, mean_states, 'b-', linewidth=2, label='Mean State')
        self.ax_stats.fill_between(time_points,
                                   np.array(mean_states) - np.array(std_states),
                                   np.array(mean_states) + np.array(std_states),
                                   alpha=0.3, color='blue', label='±1 Std Dev')
        self.ax_stats.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Initial')
        self.ax_stats.axvline(x=20, color='r', linestyle='--', alpha=0.5, label='Disturbance')
        self.ax_stats.axvline(x=self.current_time_step, color='orange', linestyle='-',
                            linewidth=2, alpha=0.7, label='Current Time')
        self.ax_stats.set_xlabel('Time', fontsize=10)
        self.ax_stats.set_ylabel('Mean State Value', fontsize=10)
        self.ax_stats.set_title('Spatial Average Dynamics', fontsize=11, fontweight='bold')
        self.ax_stats.legend(loc='best', fontsize=8)
        self.ax_stats.grid(True, alpha=0.3)

        # 更新直方图：当前状态值分布
        self.ax_hist.clear()
        self.ax_hist.hist(current_state.flatten(), bins=30, color='steelblue',
                         edgecolor='black', alpha=0.7)
        self.ax_hist.axvline(x=100, color='g', linestyle='--', linewidth=2,
                           label='Initial State')
        self.ax_hist.set_xlabel('State Value', fontsize=10)
        self.ax_hist.set_ylabel('Frequency', fontsize=10)
        self.ax_hist.set_title(f'State Distribution at t={self.current_time_step}',
                              fontsize=11, fontweight='bold')
        self.ax_hist.legend(fontsize=8)
        self.ax_hist.grid(True, alpha=0.3, axis='y')

        self.fig.canvas.draw_idle()

    def show(self):
        """显示界面"""
        plt.show()


class AnimationVisualization:
    """动画可视化"""

    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        self.setup_scenario()

    def setup_scenario(self):
        """设置场景"""
        print("\n选择预设场景:")
        print("1. 均质生态系统 + 圆形干扰")
        print("2. 异质生态系统 + 随机干扰")
        print("3. 梯度生态系统 + 梯度干扰")
        print("4. 高扩散系统 + 矩形干扰")

        choice = input("请选择 (1-4, 默认1): ").strip() or '1'

        if choice == '1':
            resistance_map = np.ones((self.grid_size, self.grid_size)) * 0.7
            resilience_map = np.ones((self.grid_size, self.grid_size)) * 0.8
            recovery_map = np.ones((self.grid_size, self.grid_size)) * 0.3
            diffusion_rate = 0.1
            disturbance_type = 'circular'

        elif choice == '2':
            # 异质性环境
            resistance_map = self.create_heterogeneous_map(0.7, 0.3)
            resilience_map = self.create_heterogeneous_map(0.8, 0.3)
            recovery_map = self.create_heterogeneous_map(0.3, 0.2)
            diffusion_rate = 0.1
            disturbance_type = 'random'

        elif choice == '3':
            # 梯度环境（从左到右退化）
            x = np.linspace(0.9, 0.3, self.grid_size)
            resistance_map = np.tile(x, (self.grid_size, 1))
            resilience_map = np.tile(x, (self.grid_size, 1))
            recovery_map = np.ones((self.grid_size, self.grid_size)) * 0.3
            diffusion_rate = 0.05
            disturbance_type = 'gradient'

        else:
            # 高扩散
            resistance_map = np.ones((self.grid_size, self.grid_size)) * 0.6
            resilience_map = np.ones((self.grid_size, self.grid_size)) * 0.8
            recovery_map = np.ones((self.grid_size, self.grid_size)) * 0.4
            diffusion_rate = 0.3
            disturbance_type = 'rectangular'

        # 创建模型
        self.model = SpatialEcosystemModel(
            grid_size=self.grid_size,
            resistance_map=resistance_map,
            resilience_map=resilience_map,
            recovery_rate_map=recovery_map,
            diffusion_rate=diffusion_rate
        )

        # 运行模拟
        self.history = self.model.simulate(
            total_time=100,
            dt=1.0,
            disturbance_time=20,
            disturbance_type=disturbance_type,
            disturbance_intensity=60
        )

    def create_heterogeneous_map(self, base_value, heterogeneity):
        """创建异质性地图"""
        x = np.linspace(0, 4 * np.pi, self.grid_size)
        y = np.linspace(0, 4 * np.pi, self.grid_size)
        X, Y = np.meshgrid(x, y)

        pattern = np.sin(X) * np.cos(Y) + 0.5 * np.sin(2 * X) * np.cos(2 * Y)
        pattern = pattern / np.max(np.abs(pattern))

        result = base_value + heterogeneity * pattern * base_value
        return np.clip(result, 0, 1)

    def animate(self):
        """创建动画"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 初始化图像
        im1 = ax1.imshow(self.history[0], cmap='RdYlGn', vmin=0, vmax=100,
                        origin='lower', interpolation='bilinear')
        ax1.set_title('Ecosystem State', fontsize=12, fontweight='bold')
        plt.colorbar(im1, ax=ax1, label='State Value')

        # 准备时间序列数据
        mean_states = [np.mean(state) for state in self.history]
        time_points = np.arange(len(self.history))

        line, = ax2.plot([], [], 'b-', linewidth=2)
        ax2.set_xlim(0, len(self.history))
        ax2.set_ylim(0, 110)
        ax2.axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Initial')
        ax2.axvline(x=20, color='r', linestyle='--', alpha=0.5, label='Disturbance')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Mean State')
        ax2.set_title('Spatial Average', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                           fontsize=12, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        def update(frame):
            im1.set_data(self.history[frame])
            line.set_data(time_points[:frame+1], mean_states[:frame+1])
            time_text.set_text(f't = {frame}')
            return im1, line, time_text

        anim = FuncAnimation(fig, update, frames=len(self.history),
                           interval=50, blit=True, repeat=True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    print("=" * 70)
    print("生态系统3R指标空间化可视化程序")
    print("=" * 70)
    print("\n可视化模式:")
    print("1. 交互式可视化 (推荐) - 可调整参数并观察空间动态")
    print("2. 动画演示 - 自动播放预设场景的时空演变")
    print("=" * 70)

    choice = input("\n请选择模式 (1/2, 默认1): ").strip() or '1'

    if choice == '1':
        print("\n正在启动交互式可视化...")
        print("提示：")
        print("- 调整滑块改变参数")
        print("- 点击'Run Simulation'重新运行模拟")
        print("- 点击'Disturbance'按钮切换干扰类型")
        print("- 拖动时间滑块查看不同时刻的空间分布")
        viz = SpatialVisualization(grid_size=50)
        viz.show()
    else:
        print("\n正在生成动画演示...")
        viz = AnimationVisualization(grid_size=50)
        viz.animate()
