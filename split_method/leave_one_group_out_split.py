import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from .cross_validation_split import load_data_list

def save_data(test_C_num:int, CV_num:int, df_train, df_valid, df_test, target:str="energy", dir_name:str="train_valid_test_cv_Eb"):
    node_feature_train_cv = [node_feature[i] for i in train_indice]
    edge_index_train_cv = [edge_index[i] for i in train_indice]
    bond_feature_train_cv = [bond_feature[i] for i in train_indice]

    node_feature_valid_cv = [node_feature[i] for i in np.arange(valid_indice[0], valid_indice[1])]
    edge_index_valid_cv = [edge_index[i] for i in np.arange(valid_indice[0], valid_indice[1])]
    bond_feature_valid_cv = [bond_feature[i] for i in np.arange(valid_indice[0], valid_indice[1])]

    node_feature_test_cv = [node_feature[i] for i in np.arange(test_indice[0],test_indice[1])]
    edge_index_test_cv = [edge_index[i] for i in np.arange(test_indice[0],test_indice[1])]
    bond_feature_test_cv = [bond_feature[i] for i in np.arange(test_indice[0],test_indice[1])]   
    
    train_data = load_data_list(node_feature_train_cv,edge_index_train_cv,
                                              bond_feature_train_cv,df_train,tar=target)

    valid_data = load_data_list(node_feature_valid_cv,edge_index_valid_cv,
                                              bond_feature_valid_cv,df_valid,tar=target)

    test_data = load_data_list(node_feature_test_cv,edge_index_test_cv,
                                              bond_feature_test_cv,df_test,tar=target)   
    
    if not os.path.exists(f'./{dir_name}'):
        os.makedirs(f'./{dir_name}')
    torch.save(train_data,f'{dir_name}/train_cv_{test_C_num}_{CV_num}.pt')
    torch.save(valid_data,f'{dir_name}/valid_cv_{test_C_num}_{CV_num}.pt')
    torch.save(test_data,f'{dir_name}/test_cv_{test_C_num}.pt') 

edge_index=torch.load("../feature/edge_index_c60.pt") 
node_feature=torch.load("../feature/node_feature_c60.pt")
edge_feature = torch.load('../feature/edge_feature_c60.pt')
df = pd.read_csv(os.path.join('../data/c20-c60-dft-all.csv'))

C_num = np.unique(df['#C'],return_counts=True)[-1][-6:][::-1]
inter = 5770
arr = []
for i,j in enumerate(C_num):
    inter -= j
    arr.append(inter)

fast = 1
slow = 0
all_com_valid = []
whole_chain = [5770] + arr
while fast <= len(arr):
    all_com_valid.append((whole_chain[fast],whole_chain[slow]))
    fast += 1
    slow += 1   
all_com_valid = all_com_valid[1:]

valid_dict = {}
C_name_list = ['C58','C56','C54','C52','C50']
for i,j in enumerate(C_name_list):
    valid_dict[all_com_valid[i]]=j

df_test = df[3958:].reset_index().drop(columns='index')
test_indice = (3958, 5770)
valid_possible = [i for i in all_com_valid if i != test_indice]
for i in range(5):
    valid_indice = valid_possible[i]
    df_valid = df[valid_indice[0]:valid_indice[1]].reset_index().drop(columns='index')
    indice = np.concatenate([np.arange(test_indice[0],test_indice[1]),np.arange(valid_indice[0], valid_indice[1])])
    train_indice = [i for i in np.arange(len(df)) if i not in indice]
    df_train = df.iloc[train_indice].reset_index().drop(columns='index')
    save_data(60,valid_dict.get(valid_indice), df_train,df_valid,df_test, target="energy", dir_name="train_valid_test_cv_Eb")