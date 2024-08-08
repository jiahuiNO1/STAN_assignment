#STAN assignment
#20240807

setwd("D:/Hatice_lab/STAN_assignment")

require(data.table)
require(plyr)
library(ggplot2)
library(dplyr)

#marker genes from DISCO ovarian cancer dataset
marker = fread("data/scRNAseq/DISCO_Marker_ovarian_cancer.csv")
min(table(marker$`Cell Type 1`))
#[1] 50
length(unique(marker$`Cell Type 1`))
#[1] 39


#mRNA ranking across 10 cell cluster for 10xGenomic scRNAseq data
cluster_rank = fread("outputs_stan/rank_genes_mRNA_scRNAseq_cell_cluster.csv")

View(cluster_rank %>% arrange(group, desc(scores)))

cluster_rank = cluster_rank %>% group_by(group) %>% slice_max(order_by = scores, n = 10)

table(cluster_rank$names %in% marker$Gene)
#FALSE  TRUE 
# 25     75

test = merge(cluster_rank, marker[,c(1,5)], by.x = "names", by.y = "Gene")
table(test$group)
#   1   2   3   4   5   6   7   8   9  10 
#  35  57 140 149  63  66 102  49  89  96

View(table(test$`Cell Type 1`[test$group == 1]))


#1: Ovarian cancer enriched cycling epithelial cell
#2: Ovarian cancer enriched S100A9+MUC16+ epithelial cell
#3: cDC2
#4: BNC2+ZFPM2+ fibroblast
#5: Ovarian cancer enriched S100A9+MUC16+ epithelial cell
#6: Ovarian cancer enriched S100A9+MUC16+ epithelial cell
#7: Plasma cell
#8: Ovarian cancer enriched S100A9+MUC16+ epithelial cell
#9: CD56 NK cell
#10: Capillary EC

