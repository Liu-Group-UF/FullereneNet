import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold


def load_data_list(node_fea,edge_index,bond_fea,df,tar="energy"):
    # get target value  
    target = {}
    target['energy'] = df['E_binding(eV)'].values.reshape(-1,1)
    
    for i, name in enumerate([ 'homo', 'lumo', 'gap',]):
        target[name] = df.iloc[:,4+i].values.reshape(-1,1)
        
    target['logP(d)'] = df['logP(dichlorobenzene)'].values.reshape(-1,1)
    
    data_list = []
    target_list={'energy':0,'homo':1,'lumo':2,'gap':3,'logP(d)':4,}
    index=int(target_list.get(tar))
    
    # load id
    id_C = [df['Cn'][i] + '_' + str(df['#iso'][i]) for i in range(len(df))]
    
    for i in tqdm(range(len(node_fea))):
        y_i = [torch.tensor(target[name][i],dtype=torch.float32) for name in ['energy', 'homo', 'lumo', 'gap',
                                                                             'logP(d)']]
        data = Data(x=node_fea[i],edge_index=edge_index[i],edge_attr=bond_fea[i],y=y_i[index],id_C=id_C[i])
        data_list.append(data)
    return data_list

def save_data(test_C_num:int, CV_num:int, target:str="energy", dir_name:str="train_valid_test_cv_Eb"):
    df_train = df.iloc[fold_indices[CV_num][0]]
    df_train = df_train.reset_index().drop(columns='index')

    df_valid = df.iloc[fold_indices[CV_num][1]]
    df_valid = df_valid.reset_index().drop(columns='index')

    df_test = df[3958:]
    df_test = df_test.reset_index().drop(columns='index')    
    
    node_feature_train_cv = [node_feature[i] for i in fold_indices[CV_num][0]]
    edge_index_train_cv = [edge_index[i] for i in fold_indices[CV_num][0]]
    bond_feature_train_cv = [bond_feature[i] for i in fold_indices[CV_num][0]]

    node_feature_valid_cv = [node_feature[i] for i in fold_indices[CV_num][1]]
    edge_index_valid_cv = [edge_index[i] for i in fold_indices[CV_num][1]]
    bond_feature_valid_cv = [bond_feature[i] for i in fold_indices[CV_num][1]]

    node_feature_test_cv = [node_feature[i] for i in indices_test]
    edge_index_test_cv = [edge_index[i] for i in indices_test]
    bond_feature_test_cv = [bond_feature[i] for i in indices_test]   
    
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

indices = np.arange(3958)
indices_test = np.arange(3958,5770)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold_indices = []

for fold, (train_index, valid_index) in enumerate(kf.split(indices)):
    fold_indices.append((train_index, valid_index))

# example for saving binding energy (Eb) data
save_data(test_C_num=60, CV_num=0, target="energy", dir_name="train_valid_test_cv_Eb")