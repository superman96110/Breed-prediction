#提取bfile文件中的未知个体，并转为raw文件格式
plink --bfile merged_left_join_fixed_renamed_qc_prune --keep unknow.txt --chr-set 29 --keep-allele-order --recode A --out unknow
