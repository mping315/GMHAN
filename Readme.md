# GMHAN


## Installation
GMHAN can be downloaded by
```shell
git clone git@github.com:mping315/GMHAN.git
```
Installation has been tested in the Linux platform

## Data
All the data we used are stored in the './data/' folder. Among them, the gene data is stored in the ./gene folder, and the miRNA data is stored in the ./mirna folder.
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