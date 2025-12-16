#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
plink 的 ref_freq.frq.strat 生成 F_matrix.tsv（SNP x CLST 的 MAF 矩阵）
要求：
- SNP 名字原样保留（chr 前缀不会动）
- 输出 SNP 顺序与 ref_freq.frq.strat 中出现顺序一致（不排序）
- 低内存：两遍扫描（第一遍收集 CLST，第二遍流式写）
输出：
  1) F_matrix.tsv
  2) snp_list.txt（与输出同顺序）
注意：
- 该脚本假设 ref_freq.frq.strat 中同一 SNP 的不同 CLST 行是连续的（plink --freq --within 通常如此）。
"""

import argparse


def detect_columns(input_file: str):
    with open(input_file, "r") as f:
        header = f.readline().strip().split()
    col_index = {name: i for i, name in enumerate(header)}
    for need in ("SNP", "CLST", "MAF"):
        if need not in col_index:
            raise ValueError(f"Input file missing column '{need}'. Header={header}")
    return col_index


def scan_all_clst(input_file: str, col_index: dict):
    snp_i = col_index["SNP"]
    clst_i = col_index["CLST"]
    maf_i = col_index["MAF"]

    clsts = set()
    with open(input_file, "r") as f:
        _ = f.readline()  # skip header
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) <= max(snp_i, clst_i, maf_i):
                continue
            clsts.add(parts[clst_i])
    return sorted(clsts)


def build_matrix_in_original_order(input_file: str, col_index: dict, clsts: list,
                                   out_matrix: str, out_snp_list: str):
    snp_i = col_index["SNP"]
    clst_i = col_index["CLST"]
    maf_i = col_index["MAF"]

    clst_to_pos = {c: i for i, c in enumerate(clsts)}

    with open(out_matrix, "w") as f_out, open(out_snp_list, "w") as f_snp:
        f_out.write("SNP\t" + "\t".join(clsts) + "\n")

        current_snp = None
        row = None  # list[float]

        def flush():
            nonlocal current_snp, row
            if current_snp is None:
                return
            f_out.write(current_snp + "\t" + "\t".join(f"{v:g}" for v in row) + "\n")
            f_snp.write(current_snp + "\n")

        with open(input_file, "r") as f:
            _ = f.readline()  # skip header
            for line in f:
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) <= max(snp_i, clst_i, maf_i):
                    continue

                snp = parts[snp_i]     # ✅ 原样保留：chr1:6085 不会变
                clst = parts[clst_i]
                maf_str = parts[maf_i]

                try:
                    maf = float(maf_str)
                except ValueError:
                    maf = 0.0

                if current_snp != snp:
                    flush()
                    current_snp = snp
                    row = [0.0] * len(clsts)

                pos = clst_to_pos.get(clst)
                if pos is not None:
                    row[pos] = maf

        flush()


def main():
    ap = argparse.ArgumentParser(description="Compute F_matrix.tsv from ref_freq.frq.strat (keep chr, keep original order).")
    ap.add_argument("--in", dest="input_file", default="ref_freq.frq.strat", help="Input .frq.strat file")
    ap.add_argument("--out", dest="out_matrix", default="F_matrix.tsv", help="Output F_matrix TSV")
    ap.add_argument("--snp_list", dest="out_snp_list", default="snp_list.txt", help="Output snp list")
    args = ap.parse_args()

    print(f"Input: {args.input_file}", flush=True)

    col_index = detect_columns(args.input_file)
    print(f"Detected columns OK. SNP={col_index['SNP']} CLST={col_index['CLST']} MAF={col_index['MAF']}", flush=True)

    print("Pass 1: scanning all CLST ...", flush=True)
    clsts = scan_all_clst(args.input_file, col_index)
    print(f"Found {len(clsts)} CLST groups: {', '.join(clsts)}", flush=True)

    print("Pass 2: streaming write F_matrix.tsv in ORIGINAL order ...", flush=True)
    build_matrix_in_original_order(args.input_file, col_index, clsts, args.out_matrix, args.out_snp_list)

    print(f"✅ Done. Wrote: {args.out_matrix} and {args.out_snp_list}", flush=True)


if __name__ == "__main__":
    main()
