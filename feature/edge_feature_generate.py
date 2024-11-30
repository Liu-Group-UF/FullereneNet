import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, rdMolDescriptors
from rdkit.Chem import rdDetermineBonds
import numpy as np
import glob
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

def get_edge_index(mol):
    adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
    edge_list = [[],[]]
    for i in range(len(adjacency_matrix)):
        idx = np.array(np.where(adjacency_matrix[i,:]==1))
        tmp = np.fix(np.ones(idx.shape[1])*i)
        edge_list[0].extend(tmp.tolist())
        edge_list[1].extend(idx[0].tolist())
    edge_list = np.array(edge_list)
    return edge_list

def find_neighbors(node_input, connected_node, all_bonds):
    '''
    node_input is the node that you need to find for neigbors; 
    connected_node is the node that connect with node_input and make a bond 
    all_bonds contains all bonds connection situation.
    '''
    bonds_contain_node = [i for i in all_bonds if i[0] == int(node_input)]
    neigbors = [i[1] for i in bonds_contain_node if i[1] != int(connected_node)]
    neigbor_1,neigbor_2 = neigbors[0],neigbors[1]
    return neigbor_1, neigbor_2

def get_bond_pair(edge_index):
    '''
    find the node pair that consist a bond
    
    '''
    start_node, end_node= edge_index[0], edge_index[1]
    all_bond=[[int(start_node[i]),int(end_node[i])] for i in range(len(start_node))]
    return all_bond

def get_distance_matrix(mol):
    dist_matrix = Chem.Get3DDistanceMatrix(mol)
    return dist_matrix

def get_bond_ring_type(mol):
    """
    Input: mol file got from rdkit by convert from xyz file
    Output: each bond' four rings types
    
    Algorithm: 
    #
    #      A1  C1    C2_a
    #      /   \    /
    #  A2-A     C-C2
    #      \   /    \
    #      0---1     C2_b
    #      /   \
    #  B2-B     D-D2
    #      \    /
    #      B1  D1
    
    """
    dist_martrix=get_distance_matrix(mol)
    edge_index=get_edge_index(mol)
    all_bonds=get_bond_pair(edge_index)
    
    ## pick bond one by one
    ## first pick bond 0-1
    edge_total=[]
    for i in all_bonds:
        edge=[]
        node_0, node_1 = i[0], i[1]
        # find the negibors of node_0, and name them node_A, node_B
        node_A, node_B = find_neighbors(node_0,node_1,all_bonds)
        node_C, node_D = find_neighbors(node_1,node_0,all_bonds)
        # we still calc distance between node A and C and node A and D, to tell which is close A, 
        # close one label C, other label D.
        if dist_martrix[node_A][node_C] > dist_martrix[node_A][node_D]:
            node_C, node_D = node_D, node_C

        ## find second level neigbors
        node_A1, node_A2 = find_neighbors(node_A,node_0,all_bonds)
        ## calc distance of node A1 to C and node A2 to C, the closer one label is A1, the onther one is A2.   
        if dist_martrix[node_A1][node_C] > dist_martrix[node_A2][node_C]:
            node_A1, node_A2 = node_A2, node_A1

        node_C1, node_C2 = find_neighbors(node_C,node_1,all_bonds)
        if dist_martrix[node_C1][node_A] > dist_martrix[node_C2][node_A]:
            node_C1, node_C2 = node_C2, node_C1    

        ##### first-ring ######
        ## we know that A-0-1-C are connected, and have 3 bonds, then we need tell whether node_A1==node_C1
        if node_A1==node_C1:
            edge.append('P')
        else: 
            edge.append('H')

        # now follow clockwise way, move to next node D, find its neighbors
        node_D1, node_D2 = find_neighbors(node_D,node_1,all_bonds)
        # node closer to B named D1, other one is D2.
        if dist_martrix[node_D1][node_B] > dist_martrix[node_D2][node_B]:
            node_D1, node_D2 = node_D2, node_D1 

        ##### second-ring ######
        # we know that C-1-D, already have 2 bonds,then we need tell whether node C2 and D2 are connected
        # find neigbors of C2
        node_C2_a,node_C2_b = find_neighbors(node_C2,node_C,all_bonds)
        if node_D2 in [node_C2_a,node_C2_b]:
            edge.append('P')
        else: 
            edge.append('H')

        ##### third-ring ######
        node_B1, node_B2 = find_neighbors(node_B,node_0,all_bonds)
        if dist_martrix[node_B1][node_D] > dist_martrix[node_B2][node_D]:
            node_B1, node_B2 = node_B2, node_B1    
        if node_B1==node_D1:
            edge.append('P')
        else: 
            edge.append('H') 

        ##### fourth-ring ######
        node_B2_a,node_B2_b = find_neighbors(node_B2,node_B,all_bonds)
        if node_A2 in [node_B2_a,node_B2_b]:
            edge.append('P')
        else: 
            edge.append('H')  
        
        edge = [''.join(edge)]

        edge_total.append(edge) 
        
    return edge_total

def map_values(input_list):
    """
    {'HHHH': 'I',
           'HHPH': 'II',
          'PHHH': 'II',
           'HPHH': 'III',
          'HHHP': 'III',
          'HPPH': 'IV',
          'PHHP': 'IV',
          'PPHH': 'IV',
          'HHPP': 'IV',
          'PHPH': 'V',
          'HPHP': 'VI',
          'HPPP': 'VII',
          'PPHP': 'VII',
          'PPPH': 'VIII',
          'PHPP': 'VIII',
          'PPPP': 'IX'}
    """
    mapping_dict={'HHHH': [1]+[0]*8, 
           'HHPH': [0]*1+[1]+[0]*7,
          'PHHH': [0]*1+[1]+[0]*7,
          'HPHH': [0]*2+[1]+[0]*6,
          'HHHP': [0]*2+[1]+[0]*6,
          'HPPH': [0]*3+[1]+[0]*5,
          'PHHP': [0]*3+[1]+[0]*5,
          'PPHH': [0]*3+[1]+[0]*5,
          'HHPP': [0]*3+[1]+[0]*5,
          'PHPH': [0]*4+[1]+[0]*4,
          'HPHP': [0]*5+[1]+[0]*3,
          'HPPP': [0]*6+[1]+[0]*2,
          'PPHP': [0]*6+[1]+[0]*2,
          'PPPH': [0]*7+[1]+[0]*1,
          'PHPP': [0]*7+[1]+[0]*1,
          'PPPP': [0]*8+[1]}
    return [mapping_dict[item[0]] for item in input_list]

def get_edge_feature(mol):
    edge_total=get_bond_ring_type(mol)
    one_hot=map_values(edge_total)
    return one_hot

# test
# file=glob.glob("./c20-c60-unopt-xyz/*.xyz")
# file=sorted(file,key=custom_sort)
# f_mol=get_mol_list(file) # mol file 
# bond_feature=[np.array(get_edge_feature(i)) for i in f_mol]