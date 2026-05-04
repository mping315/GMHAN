# GMHAN


## Installation
GMHAN can be downloaded by
```shell
git clone git@github.com:mping315/GMHAN.git
```
Installation has been tested in the Linux platform

## Data
All the data we used are stored in the './data/' folder. Among them, the gene data is stored in the ./gene folder, and the miRNA data is stored in the ./mirna folder.

./data/gene/total_feature_500.csv: We extracted multi-omics features of genes using EMOGI and node2vec. 
For each of the 16 cancer types, we calculated mutation frequency, DNA methylation difference rate, and gene differential expression value as multi-omics features. We then ran node2vec on a gene-gene network to obtain gene topological features.

./data/gene/gene_graph.bin: This file contains the gene-gene and gene-miRNA networks we built. 
The gene-gene network was taken from the CPDB database, and the gene-miRNA network was built using data from miRTarBase, TarBase, and miRWalk database.

./data/gene/gene_trainval_fold_0.txt: Sample labels for the gene training set.
./data/gene/gene_trainval_mask_fold_0.txt: Sample index for the gene training set.
./data/gene/gene_test_fold_0.txt: Sample labels for the gene test set.
./data/gene/gene_test_mask_fold_0.txt: Sample index for the gene test set.


./data/miRNA/6.feature_miRNA_index_total.csv: This file contains the miRNA features we extracted. 
For each of the 16 cancer types, we calculated the mean z-score of miRNA expression levels, the mean of miRNA expression values, and a GIP similarity matrix based on miRNA-disease associations.

./data/miRNA/miRNA_graph.bin: This file contains the miRNA-miRNA and gene-miRNA networks we built. 
The miRNA-miRNA network was constructed using Pearson similarity between miRNAs, and the gene-miRNA network was taken from miRTarBase, TarBase, and miRWalk.

./data/miRNA/mirna_trainval_fold_0.txt: Sample labels for the miRNA training set.
./data/miRNA/mirna_trainval_mask_fold_0.txt: Sample index for the miRNA training set.
./data/miRNA/mirna_test_fold_0.txt: Sample labels for the miRNA test set.
./data/miRNA/mirna_test_mask_fold_0.txt: Sample index for the miRNA test set.

## Overview
Here, we provied an implementation of GMHAN, in Pytorch and Pytorch Geometric.
- propress_deepwalk.py: Gene feature pretreatment.
- model.py: Defines the core architecture of GMHAN.
- utils.py: Functions used in GMHAN.
- train and test.ipynb: Train and evaluate the model.
## Requirements
- Python 3.7
- Pytorch 1.12.1
- Pyroch Geometric 2.0.4
- numpy 1.18.5
- pandas 1.3.5
