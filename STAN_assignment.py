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


##part 1
#adata = sc.datasets.visium_sge(sample_id="CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma",  include_hires_tiff = True)
#adata = sc.datasets.visium_sge(sample_id="V1_Human_Lymph_Node", include_hires_tiff = True)
data_dir = "data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma"
adata = sc.read_visium(data_dir)
adata.var_names_make_unique()
print(adata)
#AnnData object with n_obs × n_vars = 4674 × 18085
#    obs: 'in_tissue', 'array_row', 'array_col'
#    var: 'gene_ids', 'feature_types', 'genome'
#    uns: 'spatial'     ##unstructured annotations
#    obsm: 'spatial'    ##observation matrices

#Observations contain metadata for each cell
adata.obs.head()
print(adata.obs_names) # cell index

#Variables contain metadata for each gene
adata.var.head()
print(adata.var_names) # gene names

#uns:
adata.uns.keys()
#KeysView(AxisArrays with keys: spatial)
print(adata.uns['spatial'])
#get the shape
#spatial_data = adata.uns['spatial']
#for key, value in spatial_data.items():
#        print(f"Key: {key}, Shape: {value.shape}")

adata.uns['spatial']['CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma'].keys()
#dict_keys(['images', 'scalefactors', 'metadata'])
adata.uns['spatial']['CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma']['scalefactors'].keys()
#dict_keys(['regist_target_img_scalef', 'tissue_hires_scalef', 'tissue_lowres_scalef', 'fiducial_diameter_fullres', 'spot_diameter_fullres'])
adata.uns['spatial']['CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma']['metadata'].keys()
#dict_keys(['chemistry_description', 'software_version'])

#pixel values for high-resolution image
adata.uns['spatial']['CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma']['images']['hires'][:5,:5]

#obsm: x, y spatial coordinate
adata.obsm.keys()
print(adata.obsm['spatial'])

#the actual gene expression data are in X
adata.X[:5, :5].todense()  # Display the first 5 rows and columns
adata.X.shape
#(4674, 18085)

#take a look at the image
sc.pl.spatial(adata, show=False,  title="H&E Stain")
plt.savefig('outputs_stan/HE_stain_image.png')
#plt.show()
#plt.close()

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

#write out the processed file
os.getcwd()
#'D:\\Hatice_lab\\STAN_assignment'
adata.write("outputs_stan/ovarian_cancer_spatial_processes.h5ad")

#Loading the gene-TF prior matrix, genes in row, TF in column, binary
adata = stan.add_gene_tf_matrix(adata,
                                min_cells_proportion = 0.2,
                                min_tfs_per_gene= 5,
                                min_genes_per_tf= 10,
                                gene_tf_source="hTFtarget",
                                tf_list="humantfs",
                                source_dir="data/TF/")

#check the sizes of the matrices after filtering.
D = adata.varm['gene_tf']
print('gene-TF matrix: {} x {}'.format(D.shape[0], D.shape[1]))
#gene-TF matrix: 10739 x 250
print('min tfs associated with each gene: {}'.format(D.T.abs().sum().min()))
#min tfs associated with each gene: 5.0
print('min genes associated with each tf: {}'.format(D.abs().sum().min()))
#min genes associated with each tf: 10.0

#adata.to_df(): converts the primary expression matrix (adata.X) of the AnnData object into a pandas DataFrame.
#cell in row, gene in column
Y = adata.to_df()
print('gene-cell matrix: {} x {}'.format(Y.shape[1], Y.shape[0]))
print('min cells associated with each gene: {}'.format((Y>0).sum().min()))
print('min genes associated with each cell: {}'.format((Y>0).T.sum().min()))
#gene-cell matrix: 10739 x 4153
#min cells associated with each gene: 831
#min genes associated with each cell: 1917

#compute spatially dependent gaussian kernel matrix K
#The spatial and morphological information is stored in adata.obsm['spatial'] and adata.obsm['pixel']
stan.pixel_intensity(adata, windowsize=25)
stan.make_kernel(adata, n=250, im_feats_weight=0.05, bandwidth=0.2)
print(adata)
#AnnData object with n_obs × n_vars = 4153 × 10739
#    obs: 'in_tissue', 'array_row', 'array_col', 'n_counts', 'ncounts', 'pixel'
#    var: 'gene_ids', 'feature_types', 'genome', 'n_cells'
#    uns: 'spatial', 'tf_names'
#    obsm: 'spatial', 'pixel', 'kernel'
#    varm: 'gene_tf'
#    layers: 'raw'
#    obsp: 'pw_dist', 'kernel'

#sc.pp.normalize_total: normalize the total counts per cell
sc.pp.normalize_total(adata)
#square root transform the total count to stabilize the variance
adata.layers['scaled'] = np.sqrt(adata.to_df())

#TF activity inference using STAN
stan.assign_folds(adata, n_folds=10, random_seed=0)
stan_model = stan.Stan(adata, layer='scaled')
stan_model.fit(n_steps=5, stages=1,
              grid_search_params={'lam1':[1e-4, 1e4], 'lam2':[1e-4, 1e4]})

adata.obsm['tfa_stan'] = pd.DataFrame(stan_model.W_concat.T, index=adata.obs_names, columns=adata.uns['tf_names'])

#4153 cells, 250 TFs
adata.obsm['tfa_stan'].shape
#(4153, 250)

#Evaluate the cross-validation performance using Pearson correlation coefficient.
cor, gene_cor = stan_model.evaluate(fold=-1)
adata.obs['pred_cor_stan'] = cor
adata.var['pred_cor_stan'] = gene_cor

print(stan_model.params)
print("Spot-wise correlation: " + str(round(np.nanmedian(cor), 4)))
#{'lam1': 10000.0, 'lam2': 10000.0}
#Spot-wise correlation: 0.2738

#Comparing STAN with Ridge regression (baseline)
ridge_model = stan.Ridge(adata, layer='scaled')
ridge_model.fit(n_steps=5, stages=1,
                grid_search_params={'lam':[1e-4, 1e4]})

adata.obsm['tfa_ridge'] = pd.DataFrame(ridge_model.W_concat.T, index=adata.obs_names, columns=adata.uns['tf_names'])

cor, gene_cor = ridge_model.evaluate(fold=-1)
adata.obs['pred_cor_ridge'] = cor
adata.var['pred_cor_ridge'] = gene_cor

print(ridge_model.params)
print("Spot-wise correlation: " + str(round(np.nanmedian(cor), 4)))
#{'lam': 0.01}
#Spot-wise correlation: 0.2303

#Evaluating the cross-validation performance
#Pearson correlation coefficient between predicted and measured gene expression profiles on held-out spots
#significantly better performance than the baseline model without spatial/morphological information based on Ridge regression.
wilcoxon(adata.obs["pred_cor_stan"], adata.obs["pred_cor_ridge"],
         zero_method='wilcox', correction=False, alternative='greater')
#WilcoxonResult(statistic=8624992.0, pvalue=0.0)

aux.plot_validation(adata)
plt.savefig('outputs_stan/validation_correlation_plot.png',bbox_inches="tight")

#the performance increase correlate with the total UMI counts per spot
pearsonr(adata.obs["pred_cor_stan"]-adata.obs["pred_cor_ridge"],
         adata.obs["n_counts"])
#PearsonRResult(statistic=0.47106307508469475, pvalue=1.8319333952858868e-228)

#look at data correlation separately by n_count
obs_get = adata.obs
obs_filter1 = obs_get[obs_get['n_counts'] <= 30000]
obs_filter1.shape
#(1718, 8)
pearsonr(obs_filter1["pred_cor_stan"]-obs_filter1["pred_cor_ridge"],
         obs_filter1["n_counts"])
#PearsonRResult(statistic=0.057502546207444515, pvalue=0.017142030946163185)
obs_filter2 = obs_get[obs_get['n_counts'] > 30000]
obs_filter2.shape
#(2435, 8)
pearsonr(obs_filter2["pred_cor_stan"]-obs_filter2["pred_cor_ridge"],
         obs_filter2["n_counts"])
#PearsonRResult(statistic=0.3765419319241947, pvalue=7.11733254128082e-83)

#look at data correlation separately by ridge correlation
obs_filter1 = obs_get[obs_get["pred_cor_ridge"] <= 0.2]
obs_filter1.shape
#(906, 8)
pearsonr(obs_filter1["pred_cor_stan"]-obs_filter1["pred_cor_ridge"],
         obs_filter1["n_counts"])
#PearsonRResult(statistic=0.2972898324325807, pvalue=6.044135598626673e-20)
obs_filter2 = obs_get[obs_get["pred_cor_ridge"] >= 0.25]
obs_filter2.shape
#(822, 8)
pearsonr(obs_filter2["pred_cor_stan"]-obs_filter2["pred_cor_ridge"],
         obs_filter2["n_counts"])
#PearsonRResult(statistic=0.532353635595089, pvalue=2.4030703610624156e-61)

adata.write("outputs_stan/ovarian_cancer_stan.h5ad")


##part 2
#Identifying spatial-domain-specific TFs (cancer specific?)
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

#normalize each spot by total counts over all genes
sc.pp.normalize_total(adata_raw)
adata_raw.layers['scaled'] = np.sqrt(adata_raw.to_df())

sc.pp.normalize_total(adata_stan)
adata_stan.layers['scaled'] = np.sqrt(adata_stan.to_df())

#Clustering of STAN-predicted TF activities identifies 9 major clusters
#the corresponding spatial distribution shows highly distinct features.
adata_tfa_scaled = adata_tfa.copy()
sc.pp.scale(adata_tfa_scaled)
sc.pp.neighbors(adata_tfa_scaled, use_rep='X', n_neighbors=100)
sc.tl.umap(adata_tfa_scaled, min_dist=0.01)
sc.tl.leiden(adata_tfa_scaled, resolution=0.55)
aux.plot_umap(adata_tfa_scaled, palette='Paired')
plt.savefig('outputs_stan/cluster_umap_plot.png')

#Associating TFs with clusters
pd.value_counts(adata_tfa_scaled.obs['leiden'])
#leiden
#0    751
#1    688
#2    586
#3    543
#4    437
#5    349
#6    302
#7    288
#8    209

adata_tfa.obs['leiden'] = adata_tfa_scaled.obs['leiden'].astype('category')
sc.tl.rank_genes_groups(adata_tfa, groupby='leiden', method="wilcoxon")
sc.pl.rank_genes_groups(adata_tfa, fontsize=fontsize, n_genes=10)
plt.savefig('outputs_stan/TF_ranking_across_cluster_plot.png')

#points, edges = aux.find_edges(adata_raw)
#import importlib
#importlib.reload(aux)
points, edges = aux.find_edges(adata_tfa,'leiden', "0")
aux.plot_spatial_activity(adata_tfa, ['SREBF1', 'ATF3', 'HIF1A'], "0", points, edges)
plt.savefig('outputs_stan/Cluster0_top3_TF_spatial_plot.png')

#find fewer associations using TF mRNA expression levels directly.
adata_raw.obs['leiden'] = adata_tfa_scaled.obs['leiden'].astype('category')
sc.tl.rank_genes_groups(adata_raw, groupby='leiden', method="wilcoxon")
aux.plot_spatial_expression(adata_raw, ['SREBF1', 'ATF3', 'HIF1A'], "0", points, edges)
plt.savefig('outputs_stan/Cluster0_top3_TF_spatial_raw_plot.png')


#compare cluster 5 with the rest
adata_tfa.obs['leiden'] = adata_tfa_scaled.obs['leiden'].astype('category')

def classify_cancer(value):
    if value == 5:
        return "Normal"
    elif 0 <= value <= 4 or 6 <= value <= 8:
        return "Cancer"
    else:
        return "Unknown"

adata_tfa.obs['cancer_normal'] = pd.to_numeric(adata_tfa_scaled.obs['leiden']).apply(classify_cancer)
adata_tfa.obs['cancer_normal'] = adata_tfa.obs['cancer_normal'].astype('category')
sc.tl.rank_genes_groups(adata_tfa, groupby='cancer_normal', method="wilcoxon")
sc.pl.rank_genes_groups(adata_tfa, fontsize=fontsize, n_genes=10)
plt.savefig('outputs_stan/TF_ranking_across_cancer_normal_plot.png', bbox_inches = "tight")

df = sc.get.rank_genes_groups_df(adata_tfa, group = ["Cancer","Normal"])
df['group'].value_counts()
#group
#Cancer    250
#Normal    250
#Name: count, dtype: int64
df.to_csv('outputs_stan/rank_genes_cancer_normal.csv')

import auxiliary_stan as aux
#import importlib
importlib.reload(aux)
points, edges = aux.find_edges(adata_tfa,'cancer_normal', "Cancer")
aux.plot_spatial_activity(adata_tfa, ['ASCL1', 'JUND', 'MEF2C'], "Cancer", points, edges)
plt.savefig('outputs_stan/Cancer_top3_TF_spatial_plot.png')

#find fewer associations using TF mRNA expression levels directly.
adata_raw.obs['leiden'] = adata_tfa_scaled.obs['leiden'].astype('category')
adata_raw.obs['cancer_normal'] = pd.to_numeric(adata_tfa_scaled.obs['leiden']).apply(classify_cancer)
adata_raw.obs['cancer_normal'] = adata_raw.obs['cancer_normal'].astype('category')
sc.tl.rank_genes_groups(adata_raw, groupby='cancer_normal', method="wilcoxon")
aux.plot_spatial_expression(adata_raw, ['ASCL1', 'JUND', 'MEF2C'], "Cancer", points, edges)
plt.savefig('outputs_stan/Cancer_top3_TF_spatial_raw_plot.png')




##part 3
#cell type specific TFs
celltypes = pd.read_csv('data/scRNAseq/cell_type.csv', index_col = 0)

obs_names = np.intersect1d(celltypes.index, adata.obs_names)
adata = adata[obs_names]
celltypes = celltypes.loc[obs_names]
row_sums = celltypes.sum(axis = 1)
(row_sums == 1).all()
#False
(row_sums != 1).sum()
#3905
(np.abs(row_sums - 1) > 1e-9).sum()
#0

celltypes_scale = celltypes.divide(celltypes.sum(axis=1), axis=0)
row_sums = celltypes_scale.sum(axis = 1)
(row_sums != 1).sum()
#1265
(np.abs(row_sums - 1) > 1e-9).sum()
#0

adata_raw.obsm['celltype_major'] = celltypes
adata_stan.obsm['celltype_major'] = celltypes
adata_tfa.obsm['celltype_major'] = celltypes

#compute correlations between spot-specific TF activities / gene expressions and cell-type proportions
mat_cor_tfa, mat_cor_rna = aux.make_cor_dataframe(adata_raw, adata_tfa, celltype_label='celltype_major')
mat_cor_tfa.head()
mat_cor_rna.head()

mat_cor_tfa.loc['FOSL1']

mat_cor_tfa.to_csv('outputs_stan/mat_cor_tfa.csv')
mat_cor_rna.to_csv('outputs_stan/mat_cor_rna.csv')

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

plot_spatial_ct_tf(ct = "4", tf = "FOSL1")
plt.savefig('outputs_stan/Celltype4_TF_FOSL1_spatial_plot.png')


#model the relationship between estimated cell type proportions and TF activities using linear regression,
#and obtain the TF score for each cell type across spots.
df_ct_tf = aux.make_ct_tf_dataframe(adata_tfa, celltype_label='celltype_major')
df_ct_tf.head()
df_ct_tf.to_csv('outputs_stan/tf_score_for_each_celltype.csv')

df_filtered = df_ct_tf.query("negative_log_p_adj>2").query('coef>1.5').query("r_squared>0.6")
tf_list = df_filtered['tf'].unique()
ct_list = df_filtered['ct'].unique()
len(tf_list)
#53
len(ct_list)
#7
aux.plot_heatmap(df_ct_tf, tf_list, ct_list)
plt.savefig('outputs_stan/Celltype_TF_score_heatmap_plot.png', bbox_inches='tight')