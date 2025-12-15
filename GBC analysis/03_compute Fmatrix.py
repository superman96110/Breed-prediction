# -*- coding:utf-8 -*-

"""
F_matrix.tsv 和 snp_list.txt
适用于内存不足场景
"""

import pandas as pd

input_file = "ref_freq.frq.strat"
chunksize = 100000  # 每次读取10万行，可根据机器调整
maf_data = {}

# 第一遍扫描所有 SNP 和 CLST，记录存在的 SNP 和 CLST
all_clst = set()
all_snp = set()

print("正在扫描 SNP 和群体（CLST）……")

for chunk in pd.read_csv(input_file, delim_whitespace=True, chunksize=chunksize):
    chunk = chunk[["SNP", "CLST", "MAF"]]
    chunk["MAF"] = chunk["MAF"].fillna(0)
    all_clst.update(chunk["CLST"].unique())
    all_snp.update(chunk["SNP"].unique())

# 排序确保列顺序一致
all_clst = sorted(list(all_clst))
all_snp = sorted(list(all_snp))

# 写入表头
with open("F_matrix.tsv", "w") as f_out:
    f_out.write("SNP\t" + "\t".join(all_clst) + "\n")

# 第二遍按 SNP 分组逐个写入
print("开始逐个写入 SNP 数据到 F_matrix.tsv ……")

reader = pd.read_csv(input_file, delim_whitespace=True, chunksize=chunksize)

# 构建一个字典缓存每个 SNP 的群体 MAF
from collections import defaultdict

snp_maf_map = defaultdict(dict)

for chunk in reader:
    chunk = chunk[["SNP", "CLST", "MAF"]]
    chunk["MAF"] = chunk["MAF"].fillna(0)
    for _, row in chunk.iterrows():
        snp_maf_map[row["SNP"]][row["CLST"]] = row["MAF"]

# 逐个 SNP 写入文件
with open("F_matrix.tsv", "a") as f_out, open("snp_list.txt", "w") as snp_out:
    for snp in all_snp:
        maf_dict = snp_maf_map.get(snp, {})
        maf_values = [str(maf_dict.get(clst, 0)) for clst in all_clst]
        f_out.write(snp + "\t" + "\t".join(maf_values) + "\n")
        snp_out.write(snp + "\n")

print("低内存版本运行完成，输出文件 F_matrix.tsv 和 snp_list.txt。")

