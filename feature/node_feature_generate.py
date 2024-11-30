import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem import rdDetermineBonds
import numpy as np
import glob
import pandas as pd
from torch.optim import Adam
import torch
from tqdm import tqdm
import os

def custom_sort(file:list):
    n1=int(file.split("/")[-1].split("_")[0].split("C")[-1])
    n2=int(file.split("/")[-1].split("_")[-1].split(".")[0])
    return (n1,n2)

def get_mol_list(file:list):
    f_mol = []
    for i,j  in enumerate(file):    
        mol = Chem.MolFromXYZFile(j)
        conn_mol = Chem.Mol(mol)
        rdDetermineBonds.DetermineBonds(conn_mol,charge=0)
        f_mol.append(conn_mol)
    return f_mol    
        
def get_rings_for_atom(atom_idx,mol):
    return [ring for ring in mol.GetRingInfo().AtomRings() if atom_idx in ring]

def get_atom_rings_info(f_mol:list):
    '''
    Input: f_mol is a list of mol file
    '''
    atom_list = []
    index_list=[]
    for index, mol in enumerate(f_mol):
        if mol is None:
            atom_rings=None
            index_list.append(index)
        elif mol is not None:
            atom_rings = {atom_idx: get_rings_for_atom(atom_idx, mol) for atom_idx in range(mol.GetNumAtoms())}
        atom_list.append(atom_rings)
        
    return atom_list,index_list

def make_up_hexagon_info(atom_list):
    '''
    some atom cannot find its hexagonal rings, so need to add it manually 
    '''
    klist = range(len(atom_list))

    degree_list=[[len(atom_list[i][0]), len(atom_list[i][1]), len(atom_list[i][2])] if len(atom_list[i])
     == 3 else [len(atom_list[i][0]), len(atom_list[i][1]), 6]  for i in klist]
    
    return degree_list

def one_hot(degree_list):
    one_hot=[]
    for i in degree_list:
        count=0
        for j in i:
            if j==5:
                count+=1
        if count==3:
            value=[1,0,0,0]
        elif count==2:
            value=[0,1,0,0]
        elif count==1:
            value=[0,0,1,0]
        elif count==0:
            value=[0,0,0,1]
        one_hot.append(value)
    return one_hot

def adjacency_matrix_to_edge_index(mol):
    # get adjacency_matrix
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
    adjacency_matrix = torch.tensor(adjacency_matrix,dtype=torch.long)
    
    # Find the indices of non-zero elements
    non_zero_indices = adjacency_matrix.nonzero(as_tuple=False)

    # The first column of non_zero_indices contains source nodes
    # and the second column contains target nodes.
    source_nodes = non_zero_indices[:, 0]
    target_nodes = non_zero_indices[:, 1]

    # Stack them to get the edge_index tensor
    edge_index = torch.stack([source_nodes, target_nodes], dim=0)

    return edge_index

def generate_node_feature_and_edge_index(file:list, edge_index_name:str="edge_index.pt", 
                                         node_feature_name:str="node_feature.pt"):
    f_mol=get_mol_list(file)
    atom_list, index_list=get_atom_rings_info(f_mol)
    node_fea=[np.array(one_hot(make_up_hexagon_info(i))) for i in atom_list]
    node_feature=[torch.tensor(i,dtype=torch.float32) for i in node_fea]
    edge_index=[adjacency_matrix_to_edge_index(i) for i in f_mol]

    torch.save(edge_index, edge_index_name)
    torch.save(node_feature, node_feature_name)
    
### test ###

# file=glob.glob("./c20-c60-unopt-xyz/*.xyz")
# file=sorted(file,key=custom_sort)
# generate_node_feature(file)
