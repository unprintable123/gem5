import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import sys, os

# 创建颜色映射
colors = [
    (0.0, "#66bf26"),  # 亮绿色 (最低负载)
    (0.5, "#ead835"),  # 黄色 (中等负载)
    (1.0, "#d92e22"),  # 红色 (最高负载)
]
cmap = mcolors.LinearSegmentedColormap.from_list(
    "custom_green_yellow_red", colors
)
norm = mcolors.Normalize(vmin=0, vmax=1)


def plot_hyperx_traffic_load(log_file):
    results = {}
    with open(log_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            if "link_utilization" in line:
                key, value = line.split("=")
                value = value.split(" (")[0]
                value = float(value)
                _, src, dst, _ = key.strip().split(".")
                results[f"{src}_{dst}"] = value
    print(f"Link Utilization Data: {results}")

    horizontal_loads = []
    for row in range(4):
        for i in range(4):
            for j in range(i + 1, 4):
                src_id = row * 4 + i
                dst_id = row * 4 + j
                horizontal_loads.append(
                    (
                        (row, i, row, j),
                        (
                            results[f"r{src_id}_out_r{dst_id}"]
                            + results[f"r{dst_id}_out_r{src_id}"]
                        )
                        / 2,
                    )
                )

    vertical_loads = []
    for col in range(4):
        for i in range(4):
            for j in range(i + 1, 4):
                src_id = i * 4 + col
                dst_id = j * 4 + col
                vertical_loads.append(
                    (
                        (i, col, j, col),
                        (
                            results[f"r{src_id}_out_r{dst_id}"]
                            + results[f"r{dst_id}_out_r{src_id}"]
                        )
                        / 2,
                    )
                )

    # 创建图形和坐标轴
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.invert_yaxis()  # 让网格从上到下编号
    # ax.set_title('4x4 HyperX Network Traffic Load', fontsize=16, pad=20)

    # 隐藏坐标轴
    ax.set_xticks([])
    ax.set_yticks([])

    # 绘制节点
    for i in range(4):
        for j in range(4):
            ax.add_patch(
                patches.Rectangle(
                    (j - 0.1, i - 0.1),
                    0.2,
                    0.2,
                    facecolor="white",
                    edgecolor="black",
                    zorder=10,
                )
            )
            ax.text(
                j,
                i,
                f"({i},{j})",
                ha="center",
                va="center",
                fontsize=10,
                zorder=11,
            )

    # 绘制水平全连接（同一行的节点之间）
    for (src_row, src_col, dst_row, dst_col), load in horizontal_loads:
        color = cmap(norm(load))
        linewidth = 2 + 2 * load  # 线宽反映负载大小

        # 计算控制点以创建弧线
        y_offset = 0.3  # 弧线高度
        assert src_row == dst_row
        verts = [
            (src_col, src_row),  # 起点
            (
                (src_col + dst_col) / 2,
                src_row - y_offset * (abs(src_col - dst_col) - 1),
            ),  # 控制点
            (dst_col, dst_row),  # 终点
        ]

        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            lw=linewidth,
            zorder=int(load * 5) + 1,
        )
        ax.add_patch(patch)

        # 添加负载值标注
        mid_x = (src_col + dst_col) / 2
        mid_y = (src_row + dst_row) / 2
        if src_col < dst_col:
            mid_y -= y_offset / 2
        else:
            mid_y += y_offset / 2

    # 绘制垂直全连接（同一列的节点之间）
    for (src_row, src_col, dst_row, dst_col), load in vertical_loads:
        color = cmap(norm(load))
        linewidth = 2 + 2 * load  # 线宽反映负载大小

        # 计算控制点以创建弧线
        x_offset = 0.3  # 弧线宽度
        assert src_col == dst_col
        verts = [
            (src_col, src_row),  # 起点
            (
                src_col - x_offset * (abs(src_row - dst_row) - 1),
                (src_row + dst_row) / 2,
            ),  # 控制点
            (dst_col, dst_row),  # 终点
        ]

        codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
        path = Path(verts, codes)
        patch = patches.PathPatch(
            path,
            facecolor="none",
            edgecolor=color,
            lw=linewidth,
            zorder=int(load * 5) + 1,
        )
        ax.add_patch(patch)

        # 添加负载值标注
        mid_x = (src_col + dst_col) / 2
        mid_y = (src_row + dst_row) / 2
        if src_row < dst_row:
            mid_x += x_offset / 2
        else:
            mid_x -= x_offset / 2

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.05)
    cbar.set_label("Link Utilization", rotation=270, labelpad=10)

    plt.tight_layout()
    # plt.show()
    plt.savefig("document/plot/hyperx_4x4_traffic_load.png", dpi=300)


plot_hyperx_traffic_load(
    "/root/gem5/logs/report/bit_complement_vc16_algo3/inj_0.800/network_stats.txt"
)
