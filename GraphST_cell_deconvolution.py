import scanpy as sc
from GraphST import GraphST
from GraphST.preprocess import filter_with_overlap_gene
import torch
from GraphST.utils import project_cell_to_spot
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


# read ST data
data_dir = "data/CytAssist_11mm_FFPE_Human_Ovarian_Carcinoma"
adata = sc.read_visium(data_dir)
adata.var_names_make_unique()

# preprocessing for ST data
GraphST.preprocess(adata)

# build graph
GraphST.construct_interaction(adata)
GraphST.add_contrastive_label(adata)

# read scRNA daa
data_dir = "data/scRNAseq/filtered_feature_bc_matrix.h5"
adata_sc = sc.read_10x_h5(data_dir)
adata_sc.var_names_make_unique()

#file_path = 'data/scRNAseq/disco_ovarian_cancer_v1.0_dup.h5ad'
#adata_sc = sc.read(file_path)
#ValueError: Observations annot. `var` must have as many rows as `X` has columns (1), but has 33538 rows
#f = h5py.File(file_path, 'r')

print(adata_sc)
#AnnData object with n_obs × n_vars = 7175 × 18082
#    var: 'gene_ids', 'feature_types', 'genome'
adata_sc.X[:5, :5].todense()
print(adata_sc.obs_names)

#add cell type clusters
cell_type = pd.read_csv("data/scRNAseq/clustering_analysis/clustering/gene_expression_graphclust/clusters.csv", index_col=0)
cell_type.index.equals(adata_sc.obs.index)
adata_sc.obs['cell_type'] = cell_type['Cluster']

# preprocessing for scRNA data
GraphST.preprocess(adata_sc)

#get highly expressed genes in each cell cluster
adata_sc.obs['cell_type'] = adata_sc.obs['cell_type'].astype('category')
sc.tl.rank_genes_groups(adata_sc, groupby='cell_type', method="wilcoxon")
sc.pl.rank_genes_groups(adata_sc, fontsize=fontsize, n_genes=10)
plt.savefig('outputs_stan/mRNA_ranking_across_cell_cluster_scRNAseq_plot.png')
plt.close()

df = sc.get.rank_genes_groups_df(adata_sc, group = None)
df.to_csv('outputs_stan/rank_genes_mRNA_scRNAseq_cell_cluster.csv')


#recode cell type clusters with cell type annotation
#OCE: Ovarian cancer enriched
mapping = {
    1: 'OCE cycling epithelial cell',
    2: 'OCE S100A9+MUC16+ epithelial cell',
    3: 'cDC2',
    4: 'BNC2+ZFPM2+ fibroblast',
    5: 'OCE S100A9+MUC16+ epithelial cell',
    6: 'OCE S100A9+MUC16+ epithelial cell',
    7: 'Plasma cell',
    8: 'OCE S100A9+MUC16+ epithelial cell',
    9: 'CD56 NK cell',
    10: 'Capillary EC'
}


cell_type['Cluster'] = cell_type['Cluster'].map(mapping)
cell_type.to_csv('data/scRNAseq/scRNAseq_cell_cluster_annotation.csv')


# find overlap genes
adata, adata_sc = filter_with_overlap_gene(adata, adata_sc)
#Number of overlap genes: 1356

# get features
GraphST.get_feature(adata)


# Run device, by default, the package is implemented on 'cpu'. We recommend using GPU.
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# Train model
model = GraphST.GraphST(adata, adata_sc, epochs=1200, random_seed=50, device=device, deconvolution=True)
adata, adata_sc = model.train_map()

adata.obsm['map_matrix'].shape
#(4674, 7175)

#the single cell data dont have annotated cell types
# Project cells into spatial space
project_cell_to_spot(adata, adata_sc, retain_percent=0.15)

print(adata)

# Visualization of spatial distribution of scRNA-seq data
with mpl.rc_context({'axes.facecolor':  'black',
                     'figure.figsize': [4.5, 5]}):

         sc.pl.spatial(adata, cmap='magma',
                  # selected cell types
                  color=['1', '2', '3', '4','5', '6', '7', '8','9', '10'],
                  ncols=5, size=1.3,
                  img_key='hires',
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=0, vmax='p99.2',
                  show=True
                 )

plt.savefig('outputs_stan/cell_type_10_cluster_spatial_plot.png')

adata.write("outputs_stan/ovarian_cancer_cell_type_cluster_data.h5ad")

#write out cell type proportions for spots
keys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
obs_subset = adata.obs[keys]
obs_subset.to_csv('data/scRNAseq/cell_type.csv')

print(pd.read_csv('data/scRNAseq/cell_type.csv', index_col=0).head())
#(4674, 10)