#运行参数为python fst_eqtl_permtest.py   --eqtl eqtl.finemapped.credible   --fst 50k_fst_pos.txt   --bg all_snp.txt   --pip 0.8   --nperm 10000   --seed 1   --out_prefix fst50k_eqtl_pip0.8
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def norm_chr(x) -> str:
    """
    统一染色体命名：
    - 'chr1','Chr1','CHR1' -> '1'
    - '1' -> '1'
    - 兼容 X/Y/MT 等（去掉chr前缀后原样保留大写）
    """
    s = str(x).strip()
    s2 = s.lower()
    if s2.startswith("chr"):
        s = s[3:]
    return s.strip()

def load_two_col_sites(path: str) -> pd.DataFrame:
    """
    读取两列位点文件：chr pos
    自动兼容空格/tab分隔；无表头；允许行首/行尾空白
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, comment="#", engine="python")
    if df.shape[1] < 2:
        raise ValueError(f"{path} 列数不足2列（chr pos）")
    df = df.iloc[:, :2].copy()
    df.columns = ["CHR", "POS"]

    df["CHR"] = df["CHR"].map(norm_chr)
    df["POS"] = pd.to_numeric(df["POS"], errors="raise").astype(np.int64)

    # 去重（避免重复位点影响计数/抽样）
    df = df.drop_duplicates()
    return df

def load_sig_eqtl(eqtl_path: str, pip_thr: float) -> dict:
    """
    从 eqtl.finemapped.credible 读取并过滤 PIP > pip_thr
    返回：dict chr -> set(pos)
    """
    df = pd.read_csv(eqtl_path, sep="\t", comment="#")
    required = {"CHR", "POS", "PIP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"eQTL文件缺少列: {missing}，当前列：{list(df.columns)}")

    df["PIP"] = pd.to_numeric(df["PIP"], errors="raise")
    df = df[df["PIP"] > pip_thr].copy()

    df["CHR"] = df["CHR"].map(norm_chr)     # chr1 -> 1
    df["POS"] = pd.to_numeric(df["POS"], errors="raise").astype(np.int64)

    by_chr = defaultdict(set)
    for c, p in zip(df["CHR"].values, df["POS"].values):
        by_chr[c].add(int(p))

    return by_chr

def count_overlap(df_sites: pd.DataFrame, eqtl_by_chr: dict) -> int:
    """统计 df_sites 中有多少个位点落在显著eQTL集合里（精确 pos 匹配）"""
    hits = 0
    for c, sub in df_sites.groupby("CHR"):
        eqset = eqtl_by_chr.get(c)
        if not eqset:
            continue
        hits += sum((int(p) in eqset) for p in sub["POS"].values)
    return hits

def build_bg_pool(bg_df: pd.DataFrame) -> dict:
    """背景位点按染色体拆成 numpy array，便于快速抽样"""
    pool = {}
    for c, sub in bg_df.groupby("CHR"):
        pool[c] = sub["POS"].values.astype(np.int64)
    return pool

def sample_chr_matched(pool: dict, chr_counts: dict, rng: np.random.Generator) -> dict:
    """
    按 chr_counts 在每条染色体上无放回抽样
    返回：dict chr -> numpy array sampled positions
    """
    sampled = {}
    for c, n in chr_counts.items():
        if n <= 0:
            continue
        if c not in pool:
            raise ValueError(f"背景集合里没有染色体 {c}，但 fst 里需要在该chr抽 {n} 个")
        arr = pool[c]
        if arr.shape[0] < n:
            raise ValueError(f"背景集合 chr{c} 位点不足：需要 {n}，只有 {arr.shape[0]}")
        idx = rng.choice(arr.shape[0], size=n, replace=False)
        sampled[c] = arr[idx]
    return sampled

def count_overlap_sampled(sampled: dict, eqtl_by_chr: dict) -> int:
    """统计抽样集合与 eQTL 的命中数（点位精确匹配）"""
    hits = 0
    for c, pos_arr in sampled.items():
        eqset = eqtl_by_chr.get(c)
        if not eqset:
            continue
        hits += len(set(map(int, pos_arr.tolist())) & eqset)
    return hits

def plot_null(null_vals: np.ndarray, obs: int, out_png: str, title: str):
    plt.figure(figsize=(8, 4))
    plt.hist(null_vals, bins=40)
    plt.axvline(obs, linestyle="--")
    plt.xlabel("Overlap count (#Fst SNPs in significant eQTL)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

def main():
    ap = argparse.ArgumentParser(
        description="Permutation test: are Fst-selected SNPs enriched in significant eQTL positions?"
    )
    ap.add_argument("--eqtl", required=True, help="eqtl.finemapped.credible（含CHR/POS/PIP）")
    ap.add_argument("--fst", required=True, help="50k_fst_pos.txt（2列：chr pos，chr为纯数字）")
    ap.add_argument("--bg", required=True, help="all_snp.txt（2列：chr pos，chr为纯数字）")
    ap.add_argument("--pip", type=float, default=0.8, help="PIP阈值，默认0.8（使用 PIP > 阈值）")
    ap.add_argument("--nperm", type=int, default=10000, help="置换次数，默认10000")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_prefix", default="fst50k_eqtl_permtest")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # 1) 读显著eQTL集合（chr1 -> 1 会自动处理）
    eqtl_by_chr = load_sig_eqtl(args.eqtl, pip_thr=args.pip)
    total_eqtl = sum(len(v) for v in eqtl_by_chr.values())
    if total_eqtl == 0:
        raise ValueError(f"PIP > {args.pip} 后没有任何eQTL位点，请检查阈值或文件。")

    # 2) 读fst位点 + 背景位点（chr为纯数字也会变成字符串 '1'）
    fst_df = load_two_col_sites(args.fst)
    bg_df  = load_two_col_sites(args.bg)

    # ——可选检查：fst/bg出现但eqtl没有的chr（不影响运行，只是提示）
    fst_chr = set(fst_df["CHR"].unique())
    bg_chr = set(bg_df["CHR"].unique())
    eq_chr = set(eqtl_by_chr.keys())
    missing_in_eqtl = sorted((fst_chr | bg_chr) - eq_chr)
    if missing_in_eqtl:
        print(f"[WARN] 这些染色体在 fst/bg 中出现，但在 (PIP>{args.pip}) 的eQTL里没有：{missing_in_eqtl}")

    # 3) 观测命中数
    obs = count_overlap(fst_df, eqtl_by_chr)

    # 4) 构造 chr 匹配抽样的零分布
    chr_counts = fst_df["CHR"].value_counts().to_dict()
    pool = build_bg_pool(bg_df)

    null_vals = np.empty(args.nperm, dtype=np.int64)
    for i in range(args.nperm):
        sampled = sample_chr_matched(pool, chr_counts, rng)
        null_vals[i] = count_overlap_sampled(sampled, eqtl_by_chr)

    # 5) 单侧p值（富集：obs更大）
    p = (np.sum(null_vals >= obs) + 1) / (args.nperm + 1)
    null_mean = float(null_vals.mean())
    null_sd = float(null_vals.std(ddof=1)) if args.nperm > 1 else 0.0
    fold = obs / null_mean if null_mean > 0 else np.nan

    # 6) 输出
    out_txt = f"{args.out_prefix}.summary.txt"
    out_png = f"{args.out_prefix}.null_dist.png"
    out_null = f"{args.out_prefix}.null_values.txt"

    pd.Series(null_vals).to_csv(out_null, index=False, header=False)

    with open(out_txt, "w") as f:
        f.write(f"pip_threshold\t{args.pip}\n")
        f.write(f"n_sig_eqtl_sites\t{total_eqtl}\n")
        f.write(f"n_fst_sites\t{fst_df.shape[0]}\n")
        f.write(f"obs_overlap\t{obs}\n")
        f.write(f"null_mean\t{null_mean:.6f}\n")
        f.write(f"null_sd\t{null_sd:.6f}\n")
        f.write(f"fold_enrichment\t{fold:.6f}\n")
        f.write(f"p_value_one_sided\t{p:.6g}\n")
        f.write(f"nperm\t{args.nperm}\n")
        f.write(f"seed\t{args.seed}\n")

    title = f"P={p:.2e}, obs={obs}, null_mean={null_mean:.2f}, fold={fold:.2f}"
    plot_null(null_vals, obs, out_png, title)

    print(f"[DONE] obs={obs}, null_mean={null_mean:.3f}, fold={fold:.3f}, p={p:.3g}")
    print(f"[OUT]  {out_txt}")
    print(f"[OUT]  {out_png}")
    print(f"[OUT]  {out_null}")

if __name__ == "__main__":
    main()

