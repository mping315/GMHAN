import numpy as np
import pandas as pd
import time
import pickle
import random
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch_geometric.nn import Node2Vec
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import dropout_adj, negative_sampling, remove_self_loops,add_self_loops

#torch.load with map_location=torch.device('cpu')
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
dgl.random.seed(0)
torch.backends.cudnn.deterministic = True

ppi=pd.read_csv('/home/mengping/experiment/01.coding_noncoding_drivergene0108/3.results/2.构建网络/cpdb_edgelist_index.csv',sep=',')
ppi=ppi.iloc[:,[2,3]]

V1=ppi.iloc[:,0].tolist()
V2=ppi.iloc[:,1].tolist()
ppi_tensor=torch.tensor([V1,V2])

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print (device)

model =  Node2Vec(ppi_tensor, embedding_dim=16, walk_length=80,
                     context_size=5,  walks_per_node=10,
                     num_negative_samples=1, p=1, q=1, sparse=True).to(device)
loader = model.loader(batch_size=128, shuffle=True)

optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.001)

def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


for epoch in range(1, 501):
    print (epoch)
    loss = train()
    print (loss)

model.eval()
str_fearures = model()
torch.save(str_fearures, '/home/mengping/experiment/01.coding_noncoding_drivergene0108/3.results/3.特征/1.gene/str_fearures_500.pkl')

#torch.nn.parameter.Parameter转numpy
numpy_para = str_fearures.detach().cpu().numpy()
df = pd.DataFrame(data=numpy_para[0:,0:] )
c=[]
for i in range(16):
    c.append('node2vec_'+str(i))
df.columns=c
df.to_csv('/home/mengping/experiment/01.coding_noncoding_drivergene0108/3.results/3.特征/1.gene/node2vec_ID_500.csv',sep=',')









