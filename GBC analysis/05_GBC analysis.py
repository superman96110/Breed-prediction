#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import sys
import time
import argparse
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait


# ---------------- Progress Bar ----------------
def render_progress(done, total, start_time, bar_width=30):
    if total <= 0:
        return
    pct = done / total
    filled = int(bar_width * pct)
    bar = "█" * filled + " " * (bar_width - filled)
    elapsed = time.time() - start_time
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("inf")
    eta_str = "∞" if eta == float("inf") else f"{int(eta)}s"
    msg = f"\r[{bar}] {done}/{total} ({pct*100:5.1f}%)  elapsed:{int(elapsed)}s  eta:{eta_str}"
    sys.stdout.write(msg)
    sys.stdout.flush()


# ---------------- Worker Globals ----------------
_F = None
_threshold = None


def _init_worker(F, threshold):
    global _F, _threshold
    _F = F
    _threshold = threshold


def _solve_one(sid, y):
    global _F, _threshold
    mask = ~np.isnan(y)
    if mask.sum() == 0:
        return sid, None

    b, _, _, _ = lstsq(_F[mask], y[mask], rcond=None)

    # 非负 + 归一化 + 阈值过滤 + 再归一化（保持你原始逻辑）
    b = np.maximum(b, 0)
    s = b.sum()
    if s > 0:
        b = b / s
        b[b < _threshold] = 0
        s2 = b.sum()
        b = b / s2 if s2 > 0 else b * 0
    else:
        b[:] = 0

    return sid, b


# ---------------- Helpers ----------------
def _parse_site_id(colname: str):
    """
    raw 列名示例：chr1:6085_C / chrX:12345_T
    我们要匹配 F_matrix.tsv 的 index（你现在是 chr1:6085 这种）
    所以这里返回：chr1:6085（保留 chr）
    """
    import re
    m = re.match(r"^(chr\w+):(\d+)_", colname)
    if not m:
        return None
    # m.group(1) = chr1 / chrX / chr3 ...
    # m.group(2) = pos
    return f"{m.group(1)}:{m.group(2)}"


def main():
    ap = argparse.ArgumentParser(
        description="GBC per-individual regression (streaming raw chunks) + multiprocessing + progress"
    )
    ap.add_argument("--raw", default="unknow.raw", help="Input .raw file (default: unknow.raw)")
    ap.add_argument("--F", default="F_matrix.tsv", help="Input F_matrix TSV (default: F_matrix.tsv)")
    ap.add_argument("--out", default="GBC_results_stream.tsv", help="Output TSV (default: GBC_results_stream.tsv)")
    ap.add_argument("--threshold", type=float, default=0.02, help="Threshold (default: 0.02)")
    ap.add_argument("--workers", type=int, default=2, help="Number of processes (default: 2)")
    ap.add_argument("--max_in_flight", type=int, default=0,
                    help="Max tasks in flight (default: 2*workers). Lower to reduce memory.")
    ap.add_argument("--chunksize", type=int, default=50,
                    help="How many individuals to read per chunk from raw (default: 50)")
    ap.add_argument("--F_is_dosage", action="store_true",
                    help="If set, assume F_matrix already stores expected dosage (0..2), so do NOT multiply by 2.")
    args = ap.parse_args()

    raw_path = args.raw
    F_path = args.F
    out_path = args.out
    threshold = float(args.threshold)
    workers = max(1, int(args.workers))
    max_in_flight = args.max_in_flight if args.max_in_flight and args.max_in_flight > 0 else 2 * workers
    chunksize = max(1, int(args.chunksize))

    # -------- Step 1: Read F_matrix --------
    print("Step 1/5: reading F_matrix ...", flush=True)
    F_df = pd.read_csv(F_path, sep="\t", index_col=0)
    components = list(F_df.columns)

    # 默认：F_df 是频率 p（0..1），回归要用期望剂量 2p 去对齐 raw 的 0/1/2
    if args.F_is_dosage:
        F_used = F_df.values.astype(float)
    else:
        F_used = (2.0 * F_df.values.astype(float))

    F_index = list(F_df.index.astype(str))
    F_index_set = set(F_index)

    print(f"  F_matrix: {F_df.shape[0]} SNPs x {F_df.shape[1]} components", flush=True)

    # -------- Step 2: Read raw header only, decide which columns to load --------
    print("Step 2/5: scanning raw header to select common SNP columns ...", flush=True)
    with open(raw_path, "r") as fh:
        header = fh.readline().strip().split()

    if "IID" not in header:
        raise ValueError("raw file missing column 'IID'")

    snp_cols = []
    col_to_site = {}

    for col in header:
        if col in ("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"):
            continue
        site = _parse_site_id(col)
        if site is None:
            continue
        col_to_site[col] = site
        if site in F_index_set:
            snp_cols.append(col)

    # 这里如果还为 0，说明位点命名仍不一致（比如 F_matrix 是 1:6085 而 raw 是 chr1:6085）
    if len(snp_cols) == 0:
        # 打印一点调试信息：展示 raw 解析出的前几个 site 和 F_matrix 前几个 index，方便你一眼看出差异
        raw_sites_preview = []
        for col in header:
            if col in ("FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"):
                continue
            s = _parse_site_id(col)
            if s:
                raw_sites_preview.append(s)
            if len(raw_sites_preview) >= 5:
                break
        print("  DEBUG raw parsed sites (first 5):", raw_sites_preview, flush=True)
        print("  DEBUG F_matrix index (first 5):", F_index[:5], flush=True)
        raise ValueError("No common SNPs found between raw columns and F_matrix index (after chr:pos parsing).")

    # 共同位点按 F_matrix 的顺序排列（保证 F 与 y 对齐）
    site_to_col = {col_to_site[c]: c for c in snp_cols}
    common_sites = [s for s in F_index if s in site_to_col]
    cols_in_order = [site_to_col[s] for s in common_sites]

    idx_map = {s: i for i, s in enumerate(F_index)}
    row_idx = np.array([idx_map[s] for s in common_sites], dtype=int)
    F_common = F_used[row_idx, :]  # shape (M_common, K)

    print(f"  Common SNPs used: {len(common_sites)}", flush=True)
    print(f"  Components (K): {len(components)}", flush=True)
    print(f"  workers={workers}, max_in_flight={max_in_flight}, chunksize={chunksize}", flush=True)
    print(f"  Output: {out_path}", flush=True)

    # -------- Step 3: Count total samples (for progress) --------
    print("Step 3/5: counting samples ...", flush=True)
    with open(raw_path, "r") as fh:
        total_samples = sum(1 for _ in fh) - 1
    total_samples = max(total_samples, 0)
    print(f"  Samples to predict: {total_samples}", flush=True)

    # -------- Step 4: Write header --------
    print("Step 4/5: writing output header ...", flush=True)
    with open(out_path, "w") as f_out:
        f_out.write("IID\t" + "\t".join(components) + "\n")

    # -------- Step 5: Streaming regression --------
    print("Step 5/5: streaming regression ...", flush=True)
    start = time.time()
    done = 0

    usecols = ["IID"] + cols_in_order

    # 关键：把 -9 当缺失，否则会严重影响回归
    reader = pd.read_csv(
        raw_path,
        sep=r"\s+",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=False,
        na_values=[-9, "-9", "NA", "NaN"]
    )

    with open(out_path, "a") as f_out:
        if workers == 1:
            for chunk in reader:
                y_mat = chunk[cols_in_order].to_numpy(dtype=float)
                sids = chunk["IID"].astype(str).tolist()

                for sid, y in zip(sids, y_mat):
                    mask = ~np.isnan(y)
                    if mask.sum() == 0:
                        b = None
                    else:
                        b, _, _, _ = lstsq(F_common[mask], y[mask], rcond=None)
                        b = np.maximum(b, 0)
                        s = b.sum()
                        if s > 0:
                            b = b / s
                            b[b < threshold] = 0
                            s2 = b.sum()
                            b = b / s2 if s2 > 0 else b * 0
                        else:
                            b[:] = 0

                    if b is not None:
                        f_out.write(sid + "\t" + "\t".join(map(str, b.tolist())) + "\n")

                    done += 1
                    render_progress(done, total_samples, start)

        else:
            futures = set()

            def submit_one(ex, sid, y):
                futures.add(ex.submit(_solve_one, sid, y))

            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(F_common, threshold)
            ) as ex:

                for chunk in reader:
                    y_mat = chunk[cols_in_order].to_numpy(dtype=float)
                    sids = chunk["IID"].astype(str).tolist()

                    for sid, y in zip(sids, y_mat):
                        while len(futures) >= max_in_flight:
                            done_set, _ = wait(futures, return_when=FIRST_COMPLETED)
                            for fut in done_set:
                                futures.remove(fut)
                                rsid, b = fut.result()
                                if b is not None:
                                    f_out.write(rsid + "\t" + "\t".join(map(str, b.tolist())) + "\n")
                                done += 1
                                render_progress(done, total_samples, start)

                        submit_one(ex, str(sid), y)

                while futures:
                    done_set, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for fut in done_set:
                        futures.remove(fut)
                        rsid, b = fut.result()
                        if b is not None:
                            f_out.write(rsid + "\t" + "\t".join(map(str, b.tolist())) + "\n")
                        done += 1
                        render_progress(done, total_samples, start)

    sys.stdout.write("\n")
    print("✅ GBC analysis completed (streaming).", flush=True)


if __name__ == "__main__":
    main()

