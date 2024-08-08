#STAN assignment
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
import os

import stan
import auxiliary_stan as aux
figsize = aux.figsize
fontsize = aux.fontsize

from anndata import AnnData
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import mean_squared_error
import squidpy as sq


#rerun part 3 cell-type specific TFs using annotated cell types


##part 3 cell type specific TFs
#load ST data
data_dir = "data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma"
adata = sc.read_visium(data_dir)
adata.var_names_make_unique()
print(adata)
#AnnData object with n_obs × n_vars = 4674 × 18085
#    obs: 'in_tissue', 'array_row', 'array_col'
#    var: 'gene_ids', 'feature_types', 'genome'
#    uns: 'spatial'     ##unstructured annotations
#    obsm: 'spatial'    ##observation matrices


#QC and preprocessing
sc.pp.filter_genes(adata, min_cells=5)
sc.pp.filter_cells(adata, min_counts=5000)
adata.layers['raw'] = adata.X
adata.to_df('raw').shape
#(4153, 18048)
#convert to pandas data frame, transpose, sum across column to get total counts for each cell
adata.obs['ncounts'] = adata.to_df('raw').T.sum()
adata.obs['ncounts'].shape
#(4153,)

#load cell types
celltypes = pd.read_csv('data/scRNAseq/annotated_cell_type.csv', index_col = 0)

obs_names = np.intersect1d(celltypes.index, adata.obs_names)
#len(obs_names)
#4153
adata = adata[obs_names]
celltypes = celltypes.loc[obs_names]
row_sums = celltypes.sum(axis = 1)
(np.abs(row_sums - 1) > 1e-9).sum()
#0

#celltypes_scale = celltypes.divide(celltypes.sum(axis=1), axis=0)
#row_sums = celltypes_scale.sum(axis = 1)
#(row_sums != 1).sum()
#1265
#(np.abs(row_sums - 1) > 1e-9).sum()
#0

# input of STAN
adata_raw = sc.read_h5ad("outputs_stan/ovarian_cancer_spatial_processes.h5ad")
# output of STAN
adata_stan = sc.read_h5ad("outputs_stan/ovarian_cancer_stan.h5ad")

#Extract the inferred TF activity matrix.
adata_tfa = AnnData(
    X = adata_stan.obsm['tfa_stan'],
    obs = adata_stan.obs,
    obsm = {name: obj for (name, obj) in adata_stan.obsm.items() if "tf" not in name},
    layers = {name: obj for (name, obj) in adata_stan.obsm.items() if "tf" in name})

adata_tfa.uns = adata_stan.uns

adata_raw.obsm['celltype_major'] = celltypes
adata_stan.obsm['celltype_major'] = celltypes
adata_tfa.obsm['celltype_major'] = celltypes

#compute correlations between spot-specific TF activities / gene expressions and cell-type proportions
mat_cor_tfa, mat_cor_rna = aux.make_cor_dataframe(adata_raw, adata_tfa, celltype_label='celltype_major')
mat_cor_tfa.head()
mat_cor_rna.head()

mat_cor_tfa.to_csv('outputs_stan/mat_cor_tfa_anno.csv')
mat_cor_rna.to_csv('outputs_stan/mat_cor_rna_anno.csv')

#
mat_cor_tfa.loc['KLF5']

def plot_spatial_ct_tf(ct, tf):
    fig, axs = plt.subplots(1,3, figsize=(figsize*3,figsize), dpi=100)
    sc.pl.spatial(sq.pl.extract(adata_tfa, "celltype_major"),
                  color=ct, alpha_img=0, ax=axs[0], show=False, cmap="rocket", size=1.8,
                  legend_fontsize=fontsize)
    sc.pl.spatial(adata_tfa, color=tf, alpha_img=0, ax=axs[1],show=False, cmap="plasma",
                  legend_fontsize=fontsize, size=1.8)
    sc.pl.spatial(adata_raw, color=tf, alpha_img=0, ax=axs[2],show=False, cmap="viridis",
                  legend_fontsize=fontsize, size=1.8)

    axs[0].set_title(ct+'\n', fontsize=fontsize)
    axs[1].set_title(tf+' activity\npearson=%.4f'% mat_cor_tfa.loc[tf, ct], fontsize=fontsize)
    axs[2].set_title(tf+" mRNA expr\npearson=%.4f"% mat_cor_rna.loc[tf, ct], fontsize=fontsize)
    for i in range(3):
        axs[i].set_xlabel("")
        axs[i].set_ylabel("")
    plt.tight_layout(pad=0.6)

plot_spatial_ct_tf(ct = "OCE S100A9+MUC16+ epithelial cell", tf = "KLF5")
plt.savefig('outputs_stan/Celltype_TF_KLF5_spatial_plot.png')
plt.close()


#model the relationship between estimated cell type proportions and TF activities using linear regression,
#and obtain the TF score for each cell type across spots.
df_ct_tf = aux.make_ct_tf_dataframe(adata_tfa, celltype_label='celltype_major')
df_ct_tf.head()
df_ct_tf.to_csv('outputs_stan/tf_score_for_each_celltype_annotated.csv')

df_filtered = df_ct_tf.query("negative_log_p_adj>2").query('coef>1').query("r_squared>0.7")
tf_list = df_filtered['tf'].unique()
ct_list = df_filtered['ct'].unique()
len(tf_list)
#25
len(ct_list)
#4
aux.plot_heatmap(df_ct_tf, tf_list, ct_list)
plt.savefig('outputs_stan/Annotated_celltype_TF_score_heatmap_plot.png', bbox_inches='tight')
plt.close()
