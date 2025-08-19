import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from .cross_validation_split import load_data_list
from .leave_one_group_out_split import save_data


edge_index=torch.load("../feature/edge_index_c60.pt") 
node_feature=torch.load("../feature/node_feature_c60.pt")
edge_feature = torch.load('../feature/edge_feature_c60.pt')
df = pd.read_csv(os.path.join('../data/c20-c60-dft-all.csv'))

data_cluster = np.array([torch.sum(node_feature[i],dim=0).numpy() for i in range(len(node_feature))])

df_test = df[3958:].reset_index().drop(columns='index')
test_indice = (3958, 5770)
cluster_data = np.array([j for i,j in enumerate(data_cluster) if i not in np.arange(3958, 5770)])
indices = [i for i,j in enumerate(data_cluster) if i not in np.arange(3958, 5770)]
        # Step 1: Shuffle data and keep track of original indices
#     indices = np.arange(cluster_data.shape[0])
shuffled_data, shuffled_indices = shuffle(cluster_data, indices, random_state=42)
    
    # Apply K-means clustering
kmeans = KMeans(n_clusters=5, random_state=42,n_init="auto",init='k-means++')
kmeans.fit(shuffled_data)
    
cluster_to_original_indices = map_clusters_to_original_indices(kmeans.labels_, shuffled_indices)
    
for i in range(5):
    valid_indcie = cluster_to_original_indices.get(i)
    df_valid = df.iloc[valid_indcie].reset_index().drop(columns='index')
    train_indice = [k for k in indices if k not in valid_indcie]
    df_train = df.iloc[train_indice].reset_index().drop(columns='index')
    save_data(60, i, df_train, df_valid, df_test, target="energy", dir_name="train_valid_test_cv_Eb")