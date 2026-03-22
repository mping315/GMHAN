"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class FeatureProjector(nn.Module):
    def __init__(self, gene_dim, mirna_dim, target_dim):
        super(FeatureProjector, self).__init__()
        self.gene_proj = nn.Linear(gene_dim, target_dim)
        self.mirna_proj = nn.Linear(mirna_dim, target_dim)
        
        # 可选: 添加激活函数和批归一化
        self.activation = nn.ReLU()
        self.gene_bn = nn.BatchNorm1d(target_dim)
        self.mirna_bn = nn.BatchNorm1d(target_dim)
    
    def forward(self, gene_feats, mirna_feats):
        # 投影基因特征
        gene_projected = self.gene_proj(gene_feats)
        gene_projected = self.gene_bn(gene_projected)
        gene_projected = self.activation(gene_projected)
        
        # 投影miRNA特征
        mirna_projected = self.mirna_proj(mirna_feats)
        mirna_projected = self.mirna_bn(mirna_projected)
        mirna_projected = self.activation(mirna_projected)
        
        return gene_projected, mirna_projected

class miRNAFeatureProjector(nn.Module):
    def __init__(self, mirna_dim, target_dim):
        super(miRNAFeatureProjector, self).__init__()
        self.mirna_proj = nn.Linear(mirna_dim, target_dim).apply(init)
        
        # 可选: 添加激活函数和批归一化
        self.activation = nn.ReLU()
        self.mirna_bn = nn.BatchNorm1d(target_dim)
        
    
    def forward(self, mirna_feats):
        
        # 投影miRNA特征
        mirna_projected = self.mirna_proj(mirna_feats)
        mirna_projected = self.mirna_bn(mirna_projected)
        mirna_projected = self.activation(mirna_projected)
        
        return mirna_projected

    
    

class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size).apply(init),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False).apply(init),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                ).apply(init)
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(
        self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )
        self.predict = nn.Linear(hidden_size * num_heads[-1], out_size).apply(init)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h)
    
class HAN_GM(nn.Module):
    def __init__(self, all_meta_paths, in_size, hidden_size, out_size, num_heads,dropout):
        super(HAN_GM, self).__init__()
        self.sum_layers = nn.ModuleList()

        for i in range(0, len(all_meta_paths)):
            self.sum_layers.append(
                HAN(all_meta_paths[i], in_size[i], hidden_size[i], out_size[i], num_heads,dropout))
        # self.dtrans = nn.Sequential(nn.Linear(out_size, 100), nn.ReLU())
        # self.ptrans = nn.Sequential(nn.Linear(out_size, 400), nn.ReLU())

    def forward(self, graph, feature):
        h1 = self.sum_layers[0](graph[0], feature[0])
        h2 = self.sum_layers[1](graph[1], feature[1])
        return h1, h2

    
class MLP(nn.Module):
    def __init__(self, nfeat):
        super(MLP, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(nfeat, 32, bias=True).apply(init),
            nn.ELU(),
            nn.Linear(32, 2, bias=True),
            # 移除LogSoftmax，在forward中直接返回logits
            #nn.LogSoftmax(dim=1)
            # nn.Sigmoid())
            #nn.Softmax(dim=1)
        )
        


    def forward(self, x):
        logits = self.MLP(x)
        return logits, F.softmax(logits, dim=1)        
            
class GMHAN(nn.Module):
    def __init__(self, target_dim, all_meta_paths, in_size, hidden_size, out_size,num_heads, dropout,gene_dim=64, mirna_dim=49):
        super(GMHAN, self).__init__()
        # 特征投影模块
        self.miRNAfeature_projector = miRNAFeatureProjector(mirna_dim, target_dim)
        self.HAN_GM = HAN_GM(all_meta_paths, in_size, hidden_size, out_size,num_heads, dropout)
        self.MLP1 = MLP(out_size[0])
        self.MLP2 = MLP(out_size[1])
        #self.init_weights()  # 初始化所有参数
        
    def forward(self, graph, h, dataset_index):
        # 投影到统一维度
        mirna_proj = self.miRNAfeature_projector(h[1])
        h_proj=[h[0],mirna_proj]
        g, m= self.HAN_GM(graph, h_proj)   
        g_logits, g_probs = self.MLP1(g[dataset_index[0]])
        m_logits, m_probs = self.MLP2(m[dataset_index[1]])
        m_logits2, m_probs2 = self.MLP2(m)
        
        return g_logits, m_logits, m_logits2, g_probs, m_probs, m_probs2, g, m


    
def init(i):
    if isinstance(i, nn.Linear):
        torch.nn.init.xavier_uniform_(i.weight)

    
    
    
    
    
    
    
    
    
    
    
    
    