#将ref-panel个体提取并保存为bfile文件格式
plink --bfile merged_left_join_fixed_renamed_qc_prune --keep ref_panel_ind.txt --make-bed --chr-set 29 --keep-allele-order --out ref_panel

#计算bfile文件的freq，使用--within参数
plink --bfile ref_panel --within group.txt --freq --out ref_freq --chr-set 29 --keep-allele-order

