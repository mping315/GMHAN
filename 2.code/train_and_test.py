import pickle
import dgl
import ast
import argparse
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from model_MLP_nopredmirna import *
from utils import *
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import logging 
import datetime
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score,f1_score
import os

import warnings
warnings.filterwarnings("ignore")



def parse_args():
    parser = argparse.ArgumentParser(
        description='Train and test GMHAN')
    
    parser.add_argument('-f', '--fold', help='number of fold (default: 0)',
                        dest='fold',
                        default=0,
                        type=int
                        )
    parser.add_argument('-v', '--node2vec', help='node2vec of epochs (default: 500)',
                        dest='node2vec',
                        default=500,
                        type=int
                        )
    parser.add_argument('-td', '--target_dim', help='project target_dim (default: 64)',
                        dest='target_dim',
                        default=64,
                        type=int
                        )
    parser.add_argument('-ghs', '--gene_hiddensize', help='the hidden size of gene (default: 32)',
                        dest='ghs',
                        default=32,
                        type=int
                        )
    parser.add_argument('-mhs', '--mirna_hiddensize', help='the hidden size of miRNA (default: 32)',
                        dest='mhs',
                        default=32,
                        type=int
                        )
    parser.add_argument('-e', '--epochs', help='maximum number of epochs (default: 1000)',
                        dest='epochs',
                        default=1000,
                        type=int
                        )
    parser.add_argument('-p', '--patience', help='patience (default: 100)',
                        dest='patience',
                        default=100,
                        type=int
                        )
    parser.add_argument('-he', '--head', help='head (default: 8)',
                        dest='head',
                        default=8,
                        type=int
                        )
    parser.add_argument('-dp', '--dropout', help='the dropout rate (default: 0.25)',
                        dest='dp',
                        default=0.2,
                        type=float
                        )
    #parser.add_argument('-h', '--head', help='the GAT head (default: 8)',
    #                    dest='h',
    #                    default=8,
    #                    type=int
    #                    )
    parser.add_argument('-lr', '--learningrate', help='the learning rate (default: 0.001)',
                        dest='lr',
                        default=0.001,
                        type=float
                        )
    parser.add_argument('-wd', '--weightdecay', help='the weight decay (default: 0.0005)',
                        dest='wd',
                        default=0.0005,
                        type=float
                        )
    parser.add_argument('-seed', '--seed', help='the random seed (default: 42)',
                        dest='seed',
                        default=42,
                        type=int
                        )
    args = parser.parse_args()
    return args


def mkdir(path):
    folder=os.path.exists(path)
    
    if not folder:
        os.makedirs(path)
        print ("-----new folder-----")
        print ("-----OK-------------")
    else:
        print ("-----There is this folder!------")

def main(args):

    #torch.load with map_location=torch.device('cpu')
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed_all(args['seed'])
    np.random.seed(args['seed'])
    random.seed(args['seed'])
    dgl.random.seed(args['seed'])
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        device = torch.device('cuda:2')
        print(f'There are {torch.cuda.device_count()} GPU(s) available.')
        print('Device name:', torch.cuda.get_device_name(device))
    else:
        print('No GPU available, using CPU instead.')
    
    start_time = datetime.datetime.now()
    print (start_time)
    
    graph_path = '/home/mengping/experiment/02.coding_noncoding_drivergene0920/3.results/2.构建网络/'
    label_path = '/home/mengping/experiment/02.coding_noncoding_drivergene0920/3.results/1.正负样本集/'
    feature_path = '/home/mengping/experiment/02.coding_noncoding_drivergene0920/3.results/3.特征/'
    result_path = '/home/mengping/experiment/02.coding_noncoding_drivergene0920/3.results/4.模型/'

    results_cwd = result_path+str(args['fold'])+".KF/"
    results_cwd = results_cwd+'target_dim_'+str(args['target_dim'])
    results_cwd = results_cwd+"he_"+str(args['head'])
    results_cwd = results_cwd+'ghs_'+str(args['ghs'])
    results_cwd = results_cwd+'/mhs_'+str(args['mhs'])
    results_cwd = results_cwd+'/dp_'+str(args['dp'])
    results_cwd = results_cwd+'/lr_'+str(args['lr'])
    results_cwd = results_cwd+'/wd_'+str(args['wd'])
    mkdir(results_cwd)
                                                                                         
    #gene
    gene_meta_paths = [['gg'],['gm','mg']]
    gene_graphs,_ = dgl.data.utils.load_graphs(graph_path+'gene_graph.bin')
    gene_g = gene_graphs[0]
    gene_g = gene_g.to(device)
    #print (gene_g)

    #mirna
    mirna_meta_paths = [['mm'],['mg','gm']]
    mirna_graphs,_ = dgl.data.utils.load_graphs(graph_path+'mirna_graph.bin')
    mirna_g = mirna_graphs[0]
    mirna_g = mirna_g.to(device)
    
    #print(gene_g.ndata['feature'])

    g=[gene_g,mirna_g]

    #gene test
    with open(label_path+'/gene_test/test_ID_batch_gene.pkl', 'rb') as f:
        gene_test_mask=pickle.load(f)

    with open(label_path+'/gene_test/test_label_batch_gene.pkl', 'rb') as f:
        gene_test_label_mask=pickle.load(f)

    #mirna test
    with open(label_path+'/mirna_test/test_ID_batch_mirna.pkl', 'rb') as f:
        mirna_test_mask=pickle.load(f)

    with open(label_path+'/mirna_test/test_label_batch_mirna.pkl', 'rb') as f:
        mirna_test_label_mask=pickle.load(f)

    #gene train
    with open(label_path+'/gene_trainvalset/trainval_ID_batch_gene.pkl', 'rb') as f:
        gene_train_mask=pickle.load(f)

    with open(label_path+'/gene_trainvalset/trainval_label_batch_gene.pkl', 'rb') as f:
        gene_train_label_mask=pickle.load(f)
    
    #mirna train
    with open(label_path+'/mirna_trainvalset/trainval_ID_batch_mirna.pkl', 'rb') as f:
        mirna_train_mask=pickle.load(f)

    with open(label_path+'/mirna_trainvalset/trainval_label_batch_mirna.pkl', 'rb') as f:
        mirna_train_label_mask=pickle.load(f)

    gene_test_mask_fold1=gene_test_mask[args['fold']]
    gene_test_label_mask_fold1=gene_test_label_mask[args['fold']]
    gene_train_mask_fold1=gene_train_mask[args['fold']]
    gene_train_label_mask_fold1=gene_train_label_mask[args['fold']]

    mirna_test_mask_fold1=mirna_test_mask[args['fold']]
    mirna_test_label_mask_fold1=mirna_test_label_mask[args['fold']]
    mirna_train_mask_fold1=mirna_train_mask[args['fold']]
    mirna_train_label_mask_fold1=mirna_train_label_mask[args['fold']]

    #训练集和测试集
    gene_test_mask_fold = gene_test_mask_fold1.to(device)
    gene_test_label_mask_fold = gene_test_label_mask_fold1.to(device)
    gene_train_mask_fold = gene_train_mask_fold1.to(device)
    gene_train_label_mask_fold = gene_train_label_mask_fold1.to(device)
    mirna_test_mask_fold = mirna_test_mask_fold1.to(device)
    mirna_test_label_mask_fold = mirna_test_label_mask_fold1.to(device)
    mirna_train_mask_fold = mirna_train_mask_fold1.to(device)
    mirna_train_label_mask_fold = mirna_train_label_mask_fold1.to(device)

    train_index=[gene_train_mask_fold,mirna_train_mask_fold]
    train_label=[gene_train_label_mask_fold,mirna_train_label_mask_fold]
    test_index=[gene_test_mask_fold,mirna_test_mask_fold]
    test_label=[gene_test_label_mask_fold,mirna_test_label_mask_fold]
    all_meta_paths=[[['gg'],['gm','mg']],
                [['mm'],['mg','gm']]]
    #all_meta_paths = [['gg'], ['gm','mg'], ['mm'], ['mg','gm']]
    #特征
    gene_fea=pd.read_csv(feature_path+'1.gene/total_feature_'+str(args['node2vec'])+'.csv',sep=',',index_col=0)
    mirna_fea=pd.read_csv(feature_path+'2.miRNA/6.feautre_miRNA_index_total.csv',sep=',')
    #print (gene_fea.shape)
    #print (gene_fea)
    #特征预处理，如果是mRNA，miRNA数据集就进行log2（x+1）和Min-Max归一化，如果是DNAmethy和RPPA数据集就跳过
    def proprecess_train_test(data):
        minmax_scaler=preprocessing.MinMaxScaler()
        data_train_scaler=minmax_scaler.fit_transform(data)
        data_train_scaler_df=pd.DataFrame(data_train_scaler)
        data_train_scaler_df.columns=data.columns
        data_train_scaler_df.index=data.index
        return (data_train_scaler_df)
    
    mirna_pro_fea=proprecess_train_test(mirna_fea)
    gene_fea_tensor=torch.tensor(gene_fea.values)
    mirna_fea_tensor=torch.tensor(mirna_pro_fea.values)
    gene_fea_tensor=gene_fea_tensor.to(torch.float32)
    mirna_fea_tensor=mirna_fea_tensor.to(torch.float32)
    
    gene_fea_tensor=gene_fea_tensor.to(device)
    mirna_fea_tensor=mirna_fea_tensor.to(device)
    node_features=[gene_fea_tensor,mirna_fea_tensor]
    
    start_time = datetime.datetime.now()
    logging.info(f'Current datetime: {start_time}')
    auc = [];aupr = [];acc = [];micro = [];macro = []    
    
    target_dim=args['target_dim']
    in_size=[args['target_dim'],args['target_dim']]
    hidden_size=[args['ghs'],args['mhs']]
    out_size=[2,2]
    num_heads=[args['head']]
    dropout=args['dp']
    lr=args['lr']
    weight_decay=args['wd']
    
    model = GMHAN(target_dim=target_dim,
        all_meta_paths=all_meta_paths,
                    in_size=in_size,
                    hidden_size=hidden_size,
                    out_size=out_size,
                    num_heads=num_heads,
                    dropout=dropout).to(device)
    
    stopper = EarlyStopping(args['patience'])
    # 在模型定义后添加损失函数
    loss_fcn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(lr=lr, weight_decay=weight_decay, params=model.parameters())
    
    dataset_train_index=train_index
    dataset_train_label=train_label
    dataset_test_index=test_index
    dataset_test_label=test_label
    
    
    def main_test(model,g,node_features,test_index,test_label):
        model.eval()
        with torch.no_grad():
            g_logits, m_logits, m_logits2, g_probs, m_probs, m_probs2,g_fea, m_fea = model(g,node_features,  dataset_test_index)
            # 
            loss_g = loss_fcn(g_logits, dataset_test_label[0].reshape(-1))
            loss_m = loss_fcn(m_logits, dataset_test_label[1].reshape(-1))
            
            # 在main_test开头添加调试信息
            #print(f"Testing - g_logits shape: {g_logits.shape if 'g_logits' in locals() else 'not defined'}")
            #print(f"Testing - test_label[0] shape: {test_label[0].shape}")
            #print(f"Testing - train_label[0] shape: {dataset_train_label[0].shape}")
            
            
            
            loss = loss_g+loss_m
            
            #gene
            g_pred1 = g_logits.cpu().numpy()
            g_label = dataset_test_label[0].cpu().numpy()
            g_acc = accuracy_score(g_label, np.argmax(g_pred1, axis=1))
            g_auc = roc_auc_score(g_label, g_probs[:, 1].cpu().numpy())
            g_aupr = average_precision_score(g_label, g_probs[:, 1].cpu().numpy())
            #mirna
            m_pred1 = m_logits.cpu().numpy()
            m_label = dataset_test_label[1].cpu().numpy()
            m_acc = accuracy_score(m_label, np.argmax(m_pred1, axis=1))
            m_auc = roc_auc_score(m_label, m_probs[:, 1].cpu().numpy())
            m_aupr = average_precision_score(m_label, m_probs[:, 1].cpu().numpy())
        return loss,g_acc,g_auc,g_aupr,m_acc,m_auc,m_aupr,g_probs,m_probs,m_probs2

    gene_best_auc=0
    gene_best_aupr=0
    gene_best_acc=0
    mirna_best_auc=0
    mirna_best_aupr=0
    mirna_best_acc=0
    best_epoch=0
    g_best_prob=[]
    m_best_prob=[]
    m_best_prob2=[]
    
    for epoch in range(args['epochs']):
        #print (epoch)
        model.train()
        g_logits, m_logits, m_logits2, g_probs, m_probs, m_probs2, g_fea, m_fea = model(g,node_features,dataset_train_index)
    
        loss_g = loss_fcn(g_logits, dataset_train_label[0].reshape(-1))
        loss_m = loss_fcn(m_logits, dataset_train_label[1].reshape(-1))
        loss = loss_g+loss_m
    
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
    
        loss,g_acc,g_auc,g_aupr,m_acc,m_auc,m_aupr,g_probs,m_probs,m_probs2=main_test(model,g,node_features,  dataset_test_index,dataset_test_label)
    
        if g_auc>gene_best_auc:
            gene_best_auc = g_auc
            gene_best_aupr = g_aupr
            gene_best_acc = g_acc
            mirna_best_auc = m_auc
            mirna_best_aupr = m_aupr
            mirna_best_acc = m_acc
            best_epoch = epoch
            g_best_prob=g_probs
            m_best_prob=m_probs
            m_best_prob2=m_probs2
    print ("---------")
    print (loss, best_epoch, gene_best_auc,gene_best_aupr,gene_best_acc,mirna_best_auc,mirna_best_aupr,mirna_best_acc)
    
    g_best_prob=g_best_prob.cpu().numpy()
    m_best_prob=m_best_prob.cpu().numpy()
    m_best_prob2 = m_best_prob2.cpu().numpy()
    np.savetxt(results_cwd+'/gene_pred.csv',g_best_prob,delimiter=',')
    np.savetxt(results_cwd+'/m_pred.csv',m_best_prob,fmt='%.4f',delimiter=',')
    np.savetxt(results_cwd+'/m_pred2.csv',m_best_prob2,fmt='%.4f',delimiter=',')

    end_time = datetime.datetime.now()
    print (end_time)


if __name__ == '__main__':

    args = parse_args()
    args_dic = vars(args)
    print('args_dict', args_dic)

    main(args_dic)
    print('The Training and test is finished!')




    
    
    
    
    
    
    