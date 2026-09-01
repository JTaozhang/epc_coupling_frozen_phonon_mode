import numpy as np
from matplotlib import pyplot as plt

from ase.io import read, write

def plot_gtotal(data:np.ndarray,rc_latt=None):
    """在二维BZ平面上绘制 g_total 分布。

    参数
    ----
    data : np.ndarray
        形状应为 (N, 4) 或更大，前 2~3 列为 k_cart 坐标，最后一列为 g_total。

    说明
    ----
    - 优先尝试把点识别成规则网格，用 pcolormesh 画真正的 colormap。
    - 如果不是规则网格，则回退到 tricontourf / scatter 的方式。
    """
    data = np.asarray(data)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError("data 的形状应至少为 (N, 4)，其中前三列是 k_cart，最后一列是 g_total")

    k_cart = data[:, :3]@rc_latt.T if rc_latt is not None else data[:, :3]
    g_total = np.sqrt(data[:, -1])

    # 由于 BZ 是二维平面，默认取前两个笛卡尔分量作图
    kx = k_cart[:, 0]
    ky = k_cart[:, 1]

    fig, ax = plt.subplots(figsize=(7.2, 6.0), constrained_layout=True)

    # 尝试识别为规则网格：kx/ky 唯一值数量乘积等于总点数
    x_unique = np.unique(np.round(kx, decimals=12))
    y_unique = np.unique(np.round(ky, decimals=12))
    is_grid = (x_unique.size * y_unique.size == data.shape[0])
    print(f"Unique kx: {x_unique.size}, Unique ky: {y_unique.size}, Total points: {data.shape[0]}, Is grid: {is_grid}")
    mappable = None
    if is_grid:
        z_grid = np.full((y_unique.size, x_unique.size), np.nan, dtype=float)
        x_map = {val: i for i, val in enumerate(x_unique)}
        y_map = {val: i for i, val in enumerate(y_unique)}
        for xi, yi, gi in zip(np.round(kx, 12), np.round(ky, 12), g_total):
            z_grid[y_map[yi], x_map[xi]] = gi

        if not np.isnan(z_grid).any():
            X, Y = np.meshgrid(x_unique, y_unique)
            mappable = ax.pcolormesh(X, Y, z_grid, shading='auto', cmap='viridis')
        else:
            is_grid = False

    if not is_grid:
        print("数据点未形成规则网格，回退到散点图方式绘制 g_total 分布。")
        try:
            import matplotlib.tri as mtri
            tri = mtri.Triangulation(kx, ky)
            mappable = ax.tricontourf(tri, g_total, levels=100, cmap='viridis')
            ax.scatter(kx, ky, c=g_total, s=8, cmap='viridis', edgecolors='none', alpha=0.6)
        except Exception:
            mappable = ax.scatter(kx, ky, c=g_total, s=18, cmap='viridis', edgecolors='none')

    cbar = fig.colorbar(mappable, ax=ax)
    cbar.set_label('g_total')

    ax.set_xlabel(r'$k_x$')
    ax.set_ylabel(r'$k_y$')
    ax.set_title('g_total distribution in 2D BZ')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(False)

    return fig, ax

results=np.loadtxt("epc_gamma_subspace.dat",skiprows=1)
print("the shape of the results:",np.shape(results))
results=results[:,2:6]  
print("the shape of the results after slicing:",np.shape(results))
structure = read("POSCAR")
rc_latt = structure.cell.reciprocal()
latt = structure.cell
print("the minimal of g val:", np.min(results[:,-1]), "the max of g val:", np.max(results[:,-1]), "the average of g val:", np.mean(results[:,-1]))
# print("Reciprocal lattice vectors (columns):\n", rc_latt)
# print("Lattice vectors (columns):\n", latt)
# print("the dot between latt and rec_latt:\n", latt @ rc_latt.T)
# print(results[:,:3]@rc_latt.T,results[0,:3])
fig, ax = plot_gtotal(results, rc_latt=rc_latt)
fig.savefig("g_total_distribution.png", dpi=300)