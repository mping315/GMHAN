# GMHAN: Identification of Coding and Non-coding Driver Genes Using Heterogeneous Graph Attention Networks
## Abstract
Motivation: Accurate identification of cancer driver genes helps researchers better understand the mechanism of tumor occurrence and promote precision oncology. However, most existing methods are largely limited to utilizing gene interaction networks and single-omics data, primarily focusing on coding genes while overlooking the role of non-coding driver genes.
Results: This study proposed a novel method named GMHAN. By constructing a gene-miRNA heterogeneous network and integrating multi-omics features and topological characteristics of genes with multi-dimensional features of miRNAs, by employing the heterogeneous graph attention network (HAN), GMHAN facilitates the simultaneous detection of coding and non-coding cancer driver genes. Compared with seven existing approaches across pan-cancer and specific cancer-type data, GMHAN achieved optimal values in both ROC and AUPR, fully demonstrating its superior capability in the comprehensive and accurate identification of coding and non-coding driver genes.
## Data
The data used in this study is saved in the './data/' folder


## Requirements
GMHAN codes is baesd on Pytorch and Python and DGL library. So you will need the following packages to run.  
+ Python==3.9.21
+ torch==2.3.1
+ dgl==2.4.0
+ torch-geometric==2.6.1
+ torchvision==0.18.1

