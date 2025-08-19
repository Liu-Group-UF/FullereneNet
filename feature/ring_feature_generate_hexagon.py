from edge_feature_generate import custom_sort, get_mol_list, get_edge_index, get_bond_pair, get_bond_ring_type
import numpy as np
import glob
import os

def get_rings_for_atom(atom_idx,mol):
    return [ring for ring in mol.GetRingInfo().AtomRings() if atom_idx in ring]

def atom_rings(mol):
    atom_rings = {atom_idx: get_rings_for_atom(atom_idx, mol) for atom_idx in range(mol.GetNumAtoms())}  
    return atom_rings

def find_hexgon_ring_node(atom_rings):
    '''
    this fucntion is used to find the node consist of hexgon, but some of them are not shown in the ring found algorithm
    '''
    node_to_find = [i for i in range (len(atom_rings)) if len(atom_rings.get(i))==2]
    return node_to_find

def sort_and_delete_duplciaed_for_all_bonds(all_bonds):
    sorted_bonds = [sorted(i) for i in all_bonds]
    final_bonds = []
    for i in sorted_bonds: 
        if i not in final_bonds:
            final_bonds.append(i)
    return final_bonds

def select_bonds_by_its_node(node, sorted_bonds):
    '''
    Input: node that need to find its corresponding bonds in sorted_bonds
    Output: A list contains the other node that makes bonds besides node input 
    '''
    Another_node = [i[1] for i in sorted_bonds if i[0] == node]
    return Another_node

def assign_node_to_hexagon(mol):
    '''
    Input: mol file
    Output: all bonds can consist of hexagon
    '''
    
    # get all bond connection situation
    all_bonds = get_bond_pair(get_edge_index(mol))
    sorted_bonds = sort_and_delete_duplciaed_for_all_bonds(all_bonds)
    
    # first find each node has how many pentagon and hexgon rings 
    node_to_ring_info = atom_rings(mol)
    
    # if the hexagon rings cannot be found by rdkit mol.GetRingInfo().AtomRings(), 
    # then we need to assign it by ourself
    node_to_assign = find_hexgon_ring_node(node_to_ring_info)
    
    ring_num = int(len(node_to_assign)/6)
    if node_to_assign != 0:
        all_bonds_for_hexagon = []
        for node in node_to_assign:
            Another_node = select_bonds_by_its_node(node, sorted_bonds)
            for j in Another_node:
                if j in node_to_assign:
                    all_bonds_for_hexagon.append([node,j])
        
    return  all_bonds_for_hexagon, ring_num

def find_and_sort_hexagons(edges):
    def dfs(node, start, path, depth):
        if depth == 6:
            if start in graph[node]:
                hexagons.add(frozenset(path))
            return

        for neighbor in graph[node]:
            if neighbor not in path or (neighbor == start and depth == 5):
                dfs(neighbor, start, path + [neighbor], depth + 1)

    def sort_hexagon(hexagon):
        for start_node in hexagon:
            for next_node in graph[start_node]:
                path = [start_node, next_node]
                while len(path) < len(hexagon):
                    last_node = path[-1]
                    prev_node = path[-2]
                    next_nodes = [n for n in graph[last_node] if n != prev_node and n in hexagon]
                    if not next_nodes:
                        break
                    path.append(next_nodes[0])
                if len(path) == len(hexagon) and path[-1] in graph[start_node]:
                    return path
        return list(hexagon)  # Fallback, should not happen

    graph = {node: set() for edge in edges for node in edge}
    for edge in edges:
        graph[edge[0]].add(edge[1])
        graph[edge[1]].add(edge[0])

    hexagons = set()
    for node in graph:
        dfs(node, node, [node], 1)

    sorted_hexagons = [sort_hexagon(hexagon) for hexagon in hexagons]
    return sorted_hexagons

def get_ring_info(mol):
    ring=mol.GetRingInfo().AtomRings()
    return [list(inner_tuple) for inner_tuple in ring]

def identify_node_for_hexagon(mol):
    node_to_ring_info = atom_rings(mol)
    
    def tell_hexagon_missed_situation(node_to_ring_info):
        index = 0
        for i in range(len(node_to_ring_info)):
            if len(node_to_ring_info.get(i)) == 2:
                index = 1
                break
        if index==0:
            result=False
        else:
            result=True
        return result
    
    has_hexagon_missed = tell_hexagon_missed_situation(node_to_ring_info)
    if has_hexagon_missed == True:
        all_bonds_for_hexagon, ring_num = assign_node_to_hexagon(mol)
        sorted_hexagons = find_and_sort_hexagons(all_bonds_for_hexagon)
        pentagon_ring_info = get_ring_info(mol)
        rings = pentagon_ring_info + sorted_hexagons
    else: 
        rings = get_ring_info(mol)
        
    return rings

def select_hexagon_ring(rings):
    '''
    filter to get only hexagons 
    '''
    hexagon=[i for i in rings if len(i)==6]
    return hexagon

def get_bond_for_hexagon_ring(hexagon_ring):
    '''
    Input: node list [3, 5, 14, 15, 12, 4]
    Output: bond list [[3, 5], [5, 14], [14, 15], [15, 12], [12, 4], [4, 3]]
    '''
    def get_bond(i):
        bond=[]
        ind1=0
        ind2=1
        i=i+[i[0]] # make node connection into a circle 
        while ind2<len(i):
            bond.append([i[ind1],i[ind2]])
            ind1+=1
            ind2+=1
        return bond
    
    bonds=[get_bond(i) for i in hexagon_ring]
    return bonds

def get_all_bond_types_for_nodes_in_a_hexagon(mol):
    '''
    Input: mol files
    Output: Four rings type for each bond based on its hexagon ring's node.
    # each hexagon has 6 nodes, 6 bonds, each bond near to 4 rings. 
    '''
    # find rings first
    rings=identify_node_for_hexagon(mol)
    # filter for hexagon ring
    hexagon_ring=select_hexagon_ring(rings)
    # find all bonds for mol file 
    all_bond=get_bond_pair(get_edge_index(mol))
    # get bond's neigbor ring's type for each bond
    neigbor_rings=get_bond_ring_type(mol)
    
    hexagon_bond_neigbors=[]
    for i in get_bond_for_hexagon_ring(hexagon_ring):  
        neigbor_type_all=[]
        for j in i:
            index=all_bond.index(j) # find bond index in all bond, since all bond and neigbor_rings have same index
            neigbor_type=neigbor_rings[index]
            neigbor_type_all.append(neigbor_type)
        hexagon_bond_neigbors.append(neigbor_type_all)
        
    return hexagon_bond_neigbors  

def switch_if_third_is_p(nested_list):
    """
    For each string in each row of the nested list, if the third character is 'H',
    swap it with the first character.

    Parameters:
    nested_list (list of list of list of str): The input nested list.

    Returns:
    list: The modified nested list with specified swaps made.
    
    The third ring is always hexagon
    """
    for row in nested_list:
        for i, element in enumerate(row):
            if element[0][2] == 'P':
                # Swap the first character with the third character
                char_list = list(element[0])
                char_list[0], char_list[2] = char_list[2], char_list[0]
                row[i] = [''.join(char_list)]
    return nested_list

def concatenate_first_chars(list_of_lists):
    """
    Concatenate the first character from each string in the sub-lists of a list of lists.

    Parameters:
    list_of_lists (list of list of str): The input list of lists.

    Returns:
    list: A list containing a single string, which is the concatenation of the first character 
    from each string in the sub-lists.
    """
    concatenated_str = ''.join(sub_list[0][0] for sub_list in list_of_lists if sub_list and sub_list[0])
    return [concatenated_str]

def one_hot_encode_for_ring_features(combination):
    """
    One-hot encode a string combination into a vector based on its type (H0 to H6).

    Parameters:
    combination (str): The string combination to be one-hot encoded.

    Returns:
    list: A one-hot encoded vector representing the type of the input string.
    """
    # Define the mapping from combination types to vector indices
    type_to_index = {"H0": 0, "H1": 1, "H2-o": 2, "H2-m": 3, "H2-p": 4, "H3-o": 5, "H3-m": 6, "H3-p": 7,
                    "H4-o": 8, "H4-m": 9, "H4-p": 10, "H5": 11, "H6": 12}

    # Count the number of 'P's in the combination
    H_count = combination.count('H')

    # Determine the type based on the count of 'P's and the pattern
    if H_count == 0:
        comb_type = "H0"
    elif H_count == 1:
        comb_type = "H1"
    elif H_count == 2:
        if "HH" in combination or "PPPP" in combination:
            comb_type = "H2-o"
        elif "HPPH" in combination:
            comb_type = "H2-p"
        else:
            comb_type = "H2-m"
    elif H_count == 3:
        if "HHH" in combination or "PPP" in combination:
            comb_type = "H3-o"
        elif "HPHPH" in combination:
            comb_type = "H3-p"
        else:
            comb_type = "H3-m"
    elif H_count == 4:
        if "PP" in combination or "HHHH" in combination:
            comb_type = "H4-o"
        elif "PHHP" in combination:
            comb_type = "H4-p"
        else:
            comb_type = "H4-m"
    elif H_count == 5:
        comb_type = "H5"
    else:  
        comb_type = "H6"

    # Create the one-hot encoded vector
    one_hot_vector = [0] * len(type_to_index)
    one_hot_vector[type_to_index[comb_type]] = 1

    return one_hot_vector

def sort_and_pick_hexagon_bond_neigbors(mol):
    '''
    We can know that in hexagon case, center is a hexagon (i.e. area 3), 
    we need to identify the type of its suorrding rings
         
    #     \ (1) /
    #      \   /
    #  (4) 0---1 (2)
    #      /   \
    #     / (3) \    
    
    Accodring to our fucntion 'get_bond_ring_type', the area in (3) will always be hexagon when we put a 
    
    pentagon and try to find its neigbor rings. Therefore, we need to make sure result get from function
    'get_all_bond_types_for_nodes_in_a_hexagon' e.g. [['PPHP'], ['PPHP'], ['HPPP'], ['HPPP'], ['HPPP'], ['PPHP']], 
    the third elements in tuple 'PPHP' is always H, if not, switch elements in position 1 and 3. 
    
    Then we can pick first elements in each tuple, and concatenate it, then it is the surroding six rings by the
    center hexagon. Then we can classify the types into 13 types. They are:
    
    H0: PPPPPP 
    H1: HPPPPP, PHPPPP, PPHPPP, PPPHPP, PPPPHP, PPPPPH
    H2-o: PPPPHH, PPPHHP, PPHHPP, PHHPPP, HPPPPH, HHPPPP
    H2-m: PPPHPH, PPHPHP, PHPPPH, PHPHPP, HPPPHP, HPHPPP
    H2-p: PPHPPH, PHPPHP, HPPHPP
    H3-o: PPPHHH, PPHHHP, PHHHPP, HPPPHH, HHPPPH, HHHPPP
    H3-m: PPHPHH, PPHHPH, PHPPHH, PHPHHP, PHHPPH, PHHPHP, HPPHPH, HPPHHP, HPHPPH, HPHHPP, HHPPHP, HHPHPP
    H3-p: PHPHPH, HPHPHP
    H4-o: HHHHPP, HHHPPH, HHPPHH, HPPHHH, PHHHHP, PPHHHH
    H4-m: HHHPHP, HHPHPH, HPHHHP, HPHPHH, PHHHPH, PHPHHH
    H4-p: HHPHHP, HPHHPH, PHHPHH   
    H5: PHHHHH, HPHHHH, HHPHHH, HHHPHH, HHHHPH, HHHHHP
    H6: HHHHHH
    
    combinations_dict = {
 'PPPPPP': 'H0',
 'HPPPPP': 'H1','PHPPPP': 'H1','PPHPPP': 'H1','PPPHPP': 'H1','PPPPHP': 'H1','PPPPPH': 'H1',
 'PPPPHH': 'H2-o','PPPHHP': 'H2-o','PPHHPP': 'H2-o','PHHPPP': 'H2-o','HPPPPH': 'H2-o','HHPPPP': 'H2-o',
 'PPPHPH': 'H2-m','PPHPHP': 'H2-m','PHPPPH': 'H2-m','PHPHPP': 'H2-m','HPPPHP': 'H2-m','HPHPPP': 'H2-m',
 'PPHPPH': 'H2-p','PHPPHP': 'H2-p','HPPHPP': 'H2-p',
 'PPPHHH': 'H3-o','PPHHHP': 'H3-o','PHHHPP': 'H3-o','HPPPHH': 'H3-o','HHPPPH': 'H3-o','HHHPPP': 'H3-o',
 'PPHPHH': 'H3-m','PPHHPH': 'H3-m','PHPPHH': 'H3-m','PHPHHP': 'H3-m','PHHPPH': 'H3-m','PHHPHP': 'H3-m',
 'HPPHPH': 'H3-m','HPPHHP': 'H3-m','HPHPPH': 'H3-m','HPHHPP': 'H3-m','HHPPHP': 'H3-m','HHPHPP': 'H3-m',
 'PHPHPH': 'H3-p','HPHPHP': 'H3-p',
 'HHHHPP': 'H4-o','HHHPPH': 'H4-o','HHPPHH': 'H4-o','HPPHHH': 'H4-o','PHHHHP': 'H4-o','PPHHHH': 'H4-o',
 'HHHPHP': 'H4-m','HHPHPH': 'H4-m','HPHHHP': 'H4-m','HPHPHH': 'H4-m','PHHHPH': 'H4-m','PHPHHH': 'H4-m',
 'HHPHHP': 'H4-p','HPHHPH': 'H4-p','PHHPHH': 'H4-p',
 'PHHHHH': 'H5','HPHHHH': 'H5','HHPHHH': 'H5','HHHPHH': 'H5','HHHHPH': 'H5','HHHHHP': 'H5',
 'HHHHHH': 'H6'}
    '''
    # sort hexagon_bond_neigbors to make ring in area (3) is pentagon 
    hexagon_bond_neigbors = get_all_bond_types_for_nodes_in_a_hexagon(mol)
    hexagon_bond_neigbors = switch_if_third_is_p(hexagon_bond_neigbors)

    # pick the first element of each tuple to get five rings for center pentagon
    neigbor_rings = [concatenate_first_chars(i) for i in hexagon_bond_neigbors]
    
    # use one hot encoding to encode 13 ring types into vector
    if neigbor_rings == []: # C-20
        one_hot_vector = [[0]*13]
    else:
        one_hot_vector = np.array([one_hot_encode_for_ring_features(i[0]) for i in neigbor_rings])

    return one_hot_vector
    
def generate_ring_feature_hexagon(xyz_file, ring_feature_hexagon_name = "ring_feature_hexagon.pt"):
    f_mol=get_mol_list(xyz_file) # convert xyz files to mol files 
    ring_feature=[sort_and_pick_hexagon_bond_neigbors(mol) for mol in f_mol]
    ring_feature=[torch.tensor(i,dtype=torch.float32) for i in ring_feature]
    torch.save(ring_feature, ring_feature_hexagon_name)

### test ###
# file=glob.glob("/blue/mingjieliu/jiruijin/program/Bruce/c20-c60-unopt-xyz/*.xyz")
# file=sorted(file,key=custom_sort)
# generate_ring_feature_hexagon(file)
