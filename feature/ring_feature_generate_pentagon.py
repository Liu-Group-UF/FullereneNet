from edge_feature_generate import custom_sort, get_mol_list, get_edge_index, get_bond_pair, get_bond_ring_type
import numpy as np
import glob
import os

def get_ring_info(mol):
    ring=mol.GetRingInfo().AtomRings()
    return [list(inner_tuple) for inner_tuple in ring]

def select_pentagon_ring(rings):
    pentagon=[i for i in rings if len(i)==5]
    return pentagon

def get_bond_for_pentagon_ring(pentagon_ring):
    '''
    Input: node list [0, 1, 2, 4, 3]
    Output: bond list [[0, 1], [1, 2], [2, 4], [4, 3], [3, 0]]
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
    
    bonds=[get_bond(i) for i in pentagon_ring]
    return bonds

def get_all_bond_types_for_nodes_in_a_pentagon(mol):
    '''
    Input: mol files
    Output: Four rings type for each bond based on its pentagon ring's node.
    # each pentagon has 5 nodes, 5 bonds, each bond near to 4 rings. 
    output e.g. 
    [[['PPPP'], ['PPPP'], ['PHPP'], ['PPHP'], ['PPPH']], 
    [['PPPP'], ['PHPP'], ['PPHP'], ['PPPH'], ['PPPP']]]
    '''
    # find rings first
    rings=get_ring_info(mol)
    # filter for pentagon ring
    pentagon_ring=select_pentagon_ring(rings)
    # find all bonds for mol file 
    all_bond=get_bond_pair(get_edge_index(mol))
    # get bond's neigbor ring's type for each bond
    neigbor_rings=get_bond_ring_type(mol)
    
    pentagon_bond_neigbors=[]
    for i in get_bond_for_pentagon_ring(pentagon_ring): # 12 pentagon rings in total 
        neigbor_type_all=[]
        for j in i:
            index=all_bond.index(j) # find bond index in all bond, since all bond and neigbor_rings have same index
            neigbor_type=neigbor_rings[index]
            neigbor_type_all.append(neigbor_type)
        pentagon_bond_neigbors.append(neigbor_type_all)
        
    return pentagon_bond_neigbors

def switch_if_third_is_h(nested_list):
    """
    For each string in each row of the nested list, if the third character is 'H',
    swap it with the first character.

    Parameters:
    nested_list (list of list of list of str): The input nested list.

    Returns:
    list: The modified nested list with specified swaps made.
    """
    for row in nested_list:
        for i, element in enumerate(row):
            if element[0][2] == 'H':
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
    One-hot encode a string combination into a vector based on its type (P0 to P5).

    Parameters:
    combination (str): The string combination to be one-hot encoded.

    Returns:
    list: A one-hot encoded vector representing the type of the input string.
    """
    # Define the mapping from combination types to vector indices
    type_to_index = {"P0": 0, "P1": 1, "P2-o": 2, "P2-m": 3, "P3-o": 4, "P3-m": 5, "P4": 6, "P5": 7}

    # Count the number of 'P's in the combination
    p_count = combination.count('P')

    # Determine the type based on the count of 'P's and the pattern
    if p_count == 0:
        comb_type = "P0"
    elif p_count == 1:
        comb_type = "P1"
    elif p_count == 2:
        comb_type = "P2-o" if "PP" in combination or "HHH" in combination else "P2-m"
    elif p_count == 3:
        comb_type = "P3-o" if "PPP" in combination or "HH" in combination else "P3-m"
    elif p_count == 4:
        comb_type = "P4"
    else:  # p_count == 5
        comb_type = "P5"

    # Create the one-hot encoded vector
    one_hot_vector = [0] * len(type_to_index)
    one_hot_vector[type_to_index[comb_type]] = 1

    return one_hot_vector

def sort_and_pick_pentagon_bond_neigbors(mol):
    '''
    We can know that in pentagon case, center is a pentagon(i.e. area 3), we need to identify the type of its suorrding rings
         
    #     \ (1) /
    #      \   /
    #  (4) 0---1 (2)
    #      /   \
    #     / (3) \    
    
    Accodring to our fucntion 'get_bond_ring_type', the area in (3) will always be pentagon when we put a 
    
    pentagon and try to find its neigbor rings. Therefore, we need to make sure result get from function
    'get_all_bond_types_for_nodes_in_a_pentagon' e.g. [[['PPPP'], ['PPPP'], ['PHPP'], ['PPHP'], ['PPPH']]], the
    third elements in tuple 'PPPP' is always P, if not, switch elements in position 1 and 3. 
    
    Then we can pick first elements in each tuple, and concatenate it, then it is the surroding five rings by the
    center pentagon. Then we can classify the types into 8 types. They are:
    
    P0: HHHHH 
    P1: PHHHH, HPHHH, HHPHH, HHHPH, HHHHP
    P2-o: PPHHH, HPPHH, HHPPH, HHHPP, PHHHP
    P2-m: PHPHH, PHHPH, HPHPH, HPHHP, HHPHP
    P3-o: PPPHH, PPHHP, PHHPP, HPPPH, HHPPP
    P3-m: PPHPH, PHPHP, PHPPH, HPHPP, HPPHP
    P4: PPPPH, PPPHP, PPHPP, PHPPP, HPPPP
    P5: PPPPP
    
    combinations_dict = {
        "HHHHH": "P0",
        "PHHHH": "P1", "HPHHH": "P1", "HHPHH": "P1", "HHHPH": "P1", "HHHHP": "P1",
        "PPHHH": "P2-o", "HPPHH": "P2-o", "HHPPH": "P2-o", "HHHPP": "P2-o", "PHHHP": "P2-o",
        "PHPHH": "P2-m", "PHHPH": "P2-m", "HPHPH": "P2-m", "HPHHP": "P2-m", "HHPHP": "P2-m",
        "PPPHH": "P3-o", "PPHHP": "P3-o", "PHHPP": "P3-o", "HPPPH": "P3-o", "HHPPP": "P3-o",
        "PPHPH": "P3-m", "PHPHP": "P3-m", "PHPPH": "P3-m", "HPHPP": "P3-m", "HPPHP": "P3-m",
        "PPPPH": "P4", "PPPHP": "P4", "PPHPP": "P4", "PHPPP": "P4", "HPPPP": "P4",
        "PPPPP": "P5"
    }
    
    '''
    # sort pentagon_bond_neigbors to make ring in area (3) is pentagon 
    pentagon_bond_neigbors = get_all_bond_types_for_nodes_in_a_pentagon(mol)
    pentagon_bond_neigbors = switch_if_third_is_h(pentagon_bond_neigbors)

    # pick the first element of each tuple to get five rings for center pentagon
    neigbor_rings = [concatenate_first_chars(i) for i in pentagon_bond_neigbors]
    
    # use one hot encoding to encode 8 ring types into vector
    one_hot_vector = np.array([one_hot_encode_for_ring_features(i[0]) for i in neigbor_rings])

    return one_hot_vector

def generate_ring_feature_pentagon(xyz_file, ring_feature_name:str="ring_feature_pentagon.pt"):
    f_mol=get_mol_list(xyz_file) # mol file 
    ring_feature=[sort_and_pick_pentagon_bond_neigbors(mol) for mol in f_mol]
    ring_feature=[torch.tensor(i,dtype=torch.float32) for i in ring_feature]
    torch.save(ring_feature,ring_feature_name)

### test ###
# file=glob.glob("/blue/mingjieliu/jiruijin/program/Bruce/c20-c60-unopt-xyz/*.xyz")
# file=sorted(file,key=custom_sort)
# generate_ring_feature_pentagon(file)