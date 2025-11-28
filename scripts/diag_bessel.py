"""
diagnostic_bessel.py

诊断Spherical Bessel Harmonics的问题
分析:
1. 离散化误差
2. Basis函数覆盖
3. 能量分布
4. 预测精度

用法:
    python diagnostic_bessel.py --checkpoint /path/to/checkpoint.ckpt
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.special import spherical_jn, sph_harm
import argparse


def analyze_discretization_error():
    """分析最近邻查表带来的离散化误差"""
    print("="*60)
    print("1. 离散化误差分析")
    print("="*60)
    
    R_max = 1.0
    n_r_grid = 20  # 当前设置
    
    # 创建网格
    r_grid = np.linspace(0.01, R_max, n_r_grid)
    
    # 随机采样一些query点
    np.random.seed(42)
    r_query = np.random.uniform(0.01, R_max, 1000)
    
    # 计算最近邻索引
    r_idx = np.argmin(np.abs(r_grid[:, None] - r_query[None, :]), axis=0)
    r_nearest = r_grid[r_idx]
    
    # 误差
    r_error = np.abs(r_query - r_nearest)
    
    print(f"  r网格点数: {n_r_grid}")
    print(f"  r网格间距: {r_grid[1] - r_grid[0]:.4f}")
    print(f"  最大r误差: {r_error.max():.4f}")
    print(f"  平均r误差: {r_error.mean():.4f}")
    print(f"  相对误差 (平均): {(r_error / r_query).mean() * 100:.2f}%")
    
    # Bessel函数对r误差的敏感度
    print("\n  Bessel函数敏感度分析:")
    for n in [0, 1]:
        for k_idx in [1, 3, 5]:
            k = k_idx * np.pi / R_max
            
            # 精确值
            j_exact = spherical_jn(n, k * r_query)
            # 近似值
            j_approx = spherical_jn(n, k * r_nearest)
            
            # 相对误差
            rel_error = np.abs(j_exact - j_approx) / (np.abs(j_exact) + 1e-8)
            
            print(f"    j_{n}(k={k_idx}π): 平均相对误差 = {rel_error.mean()*100:.2f}%, "
                  f"最大 = {rel_error.max()*100:.2f}%")
    
    return r_error


def analyze_basis_coverage():
    """分析basis函数在不同区域的覆盖情况"""
    print("\n" + "="*60)
    print("2. Basis函数覆盖分析")
    print("="*60)
    
    n_max, l_max, n_k = 1, 3, 5
    R_max = 1.0
    
    # 创建测试网格
    r = np.linspace(0.01, R_max, 50)
    theta = np.linspace(0, np.pi, 30)
    phi = np.linspace(0, 2*np.pi, 60)
    
    # 计算不同r处的basis函数能量
    print("\n  不同半径处的basis能量:")
    for r_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        total_energy = 0
        for n in range(n_max + 1):
            for k_idx in range(1, n_k + 1):
                k = k_idx * np.pi / R_max
                j_val = spherical_jn(n, k * r_val)
                total_energy += j_val**2
        print(f"    r={r_val:.1f}: Σ j_n²(k·r) = {total_energy:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Bessel functions j_n(k*r)
    ax = axes[0, 0]
    for n in range(n_max + 1):
        for k_idx in [1, 3, 5]:
            k = k_idx * np.pi / R_max
            j_vals = [spherical_jn(n, k * r_val) for r_val in r]
            ax.plot(r, j_vals, label=f'j_{n}(k={k_idx}π·r)')
    ax.set_xlabel('r')
    ax.set_ylabel('j_n(k·r)')
    ax.set_title('Spherical Bessel Functions')
    ax.legend(fontsize=8)
    ax.grid(True)
    
    # Plot 2: |Y_l^m| at theta=π/2
    ax = axes[0, 1]
    theta_mid = np.pi / 2
    for l in range(l_max + 1):
        for m in [-l, 0, l]:
            if m >= -l and m <= l:
                Y_vals = [np.abs(sph_harm(m, l, p, theta_mid)) for p in phi]
                ax.plot(phi, Y_vals, label=f'|Y_{l}^{m}|')
    ax.set_xlabel('φ')
    ax.set_ylabel('|Y_l^m|')
    ax.set_title(f'Spherical Harmonics at θ=π/2')
    ax.legend(fontsize=8)
    ax.grid(True)
    
    # Plot 3: Discretization error vs r
    ax = axes[0, 2]
    n_r_grid = 20
    r_grid = np.linspace(0.01, R_max, n_r_grid)
    r_dense = np.linspace(0.01, R_max, 200)
    r_idx = np.argmin(np.abs(r_grid[:, None] - r_dense[None, :]), axis=0)
    r_nearest = r_grid[r_idx]
    r_error = np.abs(r_dense - r_nearest)
    ax.plot(r_dense, r_error)
    ax.set_xlabel('r')
    ax.set_ylabel('Discretization Error')
    ax.set_title(f'r Discretization Error (n_grid={n_r_grid})')
    ax.grid(True)
    
    # Plot 4: Basis function magnitude distribution
    ax = axes[1, 0]
    basis_mags = []
    for n in range(n_max + 1):
        for k_idx in range(1, n_k + 1):
            k = k_idx * np.pi / R_max
            for r_val in np.linspace(0.1, 0.9, 20):
                basis_mags.append(np.abs(spherical_jn(n, k * r_val)))
    ax.hist(basis_mags, bins=50)
    ax.set_xlabel('|j_n(k·r)|')
    ax.set_ylabel('Count')
    ax.set_title('Bessel Function Magnitude Distribution')
    ax.grid(True)
    
    # Plot 5: Suggested improvement - more r grid points
    ax = axes[1, 1]
    for n_grid in [10, 20, 50, 100]:
        r_grid = np.linspace(0.01, R_max, n_grid)
        r_dense = np.linspace(0.01, R_max, 500)
        r_idx = np.argmin(np.abs(r_grid[:, None] - r_dense[None, :]), axis=0)
        r_nearest = r_grid[r_idx]
        errors = np.abs(r_dense - r_nearest)
        ax.plot(r_dense, errors, label=f'n_grid={n_grid}')
    ax.set_xlabel('r')
    ax.set_ylabel('Error')
    ax.set_title('Discretization Error vs Grid Size')
    ax.legend()
    ax.grid(True)
    
    # Plot 6: Frequency content
    ax = axes[1, 2]
    for k_idx in range(1, n_k + 1):
        k = k_idx * np.pi / R_max
        j0_vals = [spherical_jn(0, k * r_val) for r_val in r]
        ax.plot(r, j0_vals, label=f'k={k_idx}π (λ={2/k_idx:.2f})')
    ax.set_xlabel('r')
    ax.set_ylabel('j_0(k·r)')
    ax.set_title('Radial Frequencies')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('bessel_analysis.png', dpi=150)
    print(f"\n  图表已保存到 bessel_analysis.png")
    
    return fig


def suggest_improvements():
    """提出改进建议"""
    print("\n" + "="*60)
    print("3. 改进建议")
    print("="*60)
    
    print("""
    方案A: 增加r网格密度
    ─────────────────────
    当前: n_r_grid = 20
    建议: n_r_grid = 50 或 100
    
    修改位置: SphericalBesselHarmonics.__init__()
    ```python
    n_r_grid: int = 50,  # 从20增加到50
    ```
    
    方案B: 使用线性插值代替最近邻
    ─────────────────────────────
    当前: basis_value = precomputed[nearest_idx]
    建议: basis_value = lerp(precomputed[idx], precomputed[idx+1], t)
    
    方案C: 直接计算Bessel函数 (用PyTorch实现)
    ──────────────────────────────────────────
    对于j_0和j_1可以用解析公式:
    - j_0(x) = sin(x)/x = sinc(x/π)
    - j_1(x) = sin(x)/x² - cos(x)/x
    
    这样可以精确计算，不需要查表。
    
    方案D: 混合架构
    ───────────────
    保持MLP(obs, r)的条件输入结构，
    但用Bessel基函数增强角度部分。
    
    方案E: 调整参数
    ───────────────
    - 增加n_k (径向频率数): 5 -> 8
    - 增加scale_factor: 10 -> 50
    - 调整学习率
    """)


def compare_with_spherical_harmonics():
    """对比两种方法的理论差异"""
    print("\n" + "="*60)
    print("4. 与Spherical Harmonics的理论对比")
    print("="*60)
    
    print("""
    Spherical Harmonics (SH):
    ─────────────────────────
    架构: MLP(obs, r) → w_{lm}(r) → Σ w_{lm}(r) · Y_l^m(θ,φ)
    
    - r作为MLP条件输入
    - 每个r都有独立的(L+1)²=16个系数
    - 相当于: 对每个r切片，独立拟合球面函数
    
    Spherical Bessel Harmonics (SBH):
    ──────────────────────────────────
    架构: MLP(obs) → w_{nklm} → Σ w_{nklm} · j_n(k·r) · Y_l^m(θ,φ)
    
    - r通过Bessel函数编码
    - 所有r共享同一组320个系数
    - 相当于: 用固定的3D基函数组合拟合整个球体
    
    关键差异:
    ──────────
    SH优势:
    1. r方向有无限自由度 (MLP可以学任意r→系数映射)
    2. 没有离散化误差
    3. 更直接的训练信号
    
    SBH优势:
    1. 理论上更优雅 (完整的3D正交基)
    2. 系数更紧凑 (320 vs 每个r独立的16)
    3. 天然的3D结构归纳偏置
    
    SBH劣势:
    1. 离散化误差
    2. 径向表达能力受限于basis数量
    3. Bessel函数振荡可能导致优化困难
    """)


def run_inference_analysis(checkpoint_path=None):
    """分析模型推理结果"""
    if checkpoint_path is None:
        print("\n跳过推理分析 (未提供checkpoint)")
        return
    
    print("\n" + "="*60)
    print("5. 推理结果分析")
    print("="*60)
    
    # TODO: 加载checkpoint并分析
    # - 能量分布
    # - 预测vs真实action的误差
    # - 哪些区域误差大
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    
    # 运行分析
    analyze_discretization_error()
    analyze_basis_coverage()
    suggest_improvements()
    compare_with_spherical_harmonics()
    run_inference_analysis(args.checkpoint)
    
    print("\n" + "="*60)
    print("分析完成!")
    print("="*60)