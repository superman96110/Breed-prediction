import pandas as pd
import numpy as np
from numpy.linalg import lstsq

# Step 1: 读取 .raw 文件
raw_df = pd.read_csv("sheep_test_allsnp.raw", sep='\s+')

# 用 IID 作为样本名
sample_ids = raw_df["IID"]

# 取 SNP 列
geno_df = raw_df.drop(columns=["FID", "IID", "PAT", "MAT", "SEX", "PHENOTYPE"]).copy()

# ✅ 修正列名：从 chr1:7934_A → 1:7934
# 支持 chrX:1234_T 等格式
cleaned_cols = geno_df.columns.str.extract(r'chr(\w+):(\d+)_')[0] + ":" + geno_df.columns.str.extract(r'chr(\w+):(\d+)_')[1]
geno_df.columns = cleaned_cols

# Step 2: 转置为 SNP × Sample 格式
geno_df = geno_df.T
geno_df.columns = sample_ids.values

# Step 3: 读取 F_matrix
F_matrix = pd.read_csv("F_matrix.tsv", sep="\t", index_col=0)

# Step 4: 对齐 SNPs
common_snps = geno_df.index.intersection(F_matrix.index)
geno_df = geno_df.loc[common_snps]
F_matrix = F_matrix.loc[common_snps]

# Step 5: GBC OLS 回归
results = {}
F = F_matrix.values  # M x N
threshold = 0.02  # 自定义阈值

for sample in geno_df.columns:
    y = geno_df[sample].values.astype(float)

    mask = ~np.isnan(y)
    y_valid = y[mask]
    F_valid = F[mask]

    if len(y_valid) == 0:
        print(f"⚠ No valid SNPs for {sample}")
        continue

    b, _, _, _ = lstsq(F_valid, y_valid, rcond=None)

    b = np.maximum(b, 0)
    if b.sum() > 0:
        b = b / b.sum()
        b[b < threshold] = 0
        b = b / b.sum()
    else:
        b[:] = 0

    results[sample] = b

# Step 6: 输出结果
result_df = pd.DataFrame(results, index=F_matrix.columns)
result_df.to_csv("GBC_results_test_allsnp.tsv", sep="\t")

print("✅ GBC 分析完成，结果保存在 GBC_results.tsv")

