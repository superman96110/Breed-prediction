#绘制更符合要求的图片
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def read_summary(path: str):
    """
    读取 summary.txt（key \t value）
    返回 dict
    """
    d = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            k, v = line.split("\t", 1)
            d[k] = v
    # 强转需要的字段
    obs = int(float(d.get("obs_overlap")))
    p = float(d.get("p_value_one_sided"))
    return obs, p, d

def kde_fill(ax, values, color, alpha=0.25, linewidth=2.5, gridsize=400):
    """
    画 KDE 曲线 + 填充（不指定具体风格，尽量接近你图）
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        raise ValueError("null_values 太少，无法做KDE")

    kde = gaussian_kde(values)

    xmin = values.min()
    xmax = values.max()
    # 给左右留白，接近示例图那种“尾部空间”
    pad = 0.10 * (xmax - xmin if xmax > xmin else 1.0)
    x = np.linspace(xmin - pad, xmax + pad, gridsize)
    y = kde(x)

    ax.plot(x, y, color=color, linewidth=linewidth)
    ax.fill_between(x, 0, y, color=color, alpha=alpha)
    ax.set_ylim(bottom=0)

def plot_one(ax, null_path, summary_path, color, xlabel, title=None):
    null_vals = np.loadtxt(null_path)
    obs, p, d = read_summary(summary_path)

    kde_fill(ax, null_vals, color=color, alpha=0.25, linewidth=2.5)

    # 观测值竖虚线
    ax.axvline(obs, color=color, linestyle="--", linewidth=2.5, dashes=(6, 6))

    # 右上角P值标注（接近你图）
    ax.text(
        0.97, 0.92,
        f"$P$ = {p:.1e}",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=14
    )

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel("Frequency", fontsize=16)
    ax.tick_params(axis="both", labelsize=13)

    if title:
        ax.set_title(title, fontsize=14)

def main():
    ap = argparse.ArgumentParser(description="Plot permutation null distribution as KDE style (like your example).")
    ap.add_argument("--null1", required=True, help="panel1 null_values.txt")
    ap.add_argument("--sum1", required=True, help="panel1 summary.txt")
    ap.add_argument("--null2", default=None, help="panel2 null_values.txt (optional)")
    ap.add_argument("--sum2", default=None, help="panel2 summary.txt (optional)")
    ap.add_argument("--out", default="perm_kde.png", help="output image")
    ap.add_argument("--xlabel1", default="Overlap", help="x label for panel1")
    ap.add_argument("--xlabel2", default="Overlap", help="x label for panel2")
    args = ap.parse_args()

    if (args.null2 is None) != (args.sum2 is None):
        raise ValueError("panel2 需要同时提供 --null2 和 --sum2，或都不提供")

    if args.null2 is None:
        # 单图
        fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
        plot_one(ax, args.null1, args.sum1, color="#1f77b4", xlabel=args.xlabel1)
    else:
        # 双panel（上下两幅）
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharey=False)
        plot_one(ax1, args.null1, args.sum1, color="#1f77b4", xlabel=args.xlabel1)
        plot_one(ax2, args.null2, args.sum2, color="#ff7f7f", xlabel=args.xlabel2)
        plt.tight_layout(h_pad=2.0)

    fig.tight_layout()
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"[DONE] saved: {args.out}")

if __name__ == "__main__":
    main()
