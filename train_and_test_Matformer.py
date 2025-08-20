import os 
import glob
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import random

import torch
from torch.optim.lr_scheduler import StepLR, OneCycleLR, CosineAnnealingWarmRestarts
from torch.optim import Adam, AdamW

from model.Matformer import Matformer
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
    
# load data to torch_geometric
def load_data_list(node_fea,edge_index, edge_fea, df, tar="Eb"):
    # get target value 
    target = {}
    target['Eb'] = df['E_binding(eV)'].values.reshape(-1,1)
    
    for i, name in enumerate([ 'homo', 'lumo', 'gap',]):
        target[name] = df.iloc[:, 4+i].values.reshape(-1,1)
        
    target['logP'] = df['logP(dichlorobenzene)'].values.reshape(-1,1)
    
    data_list = []
    target_list={'Eb':0,'homo':1,'lumo':2,'gap':3,'logP':4,}
    index=int(target_list.get(tar))
    
    # load id
    id_C = df['#iso']
    
    for i in tqdm(range(len(node_fea))):
        y_i = [torch.tensor(target[name][i],dtype=torch.float32) for name in ['Eb', 'homo', 'lumo', 'gap',
                                                                             'logP']]
        data = Data(x=node_fea[i],edge_index=edge_index[i],edge_attr=edge_fea[i],y=y_i[index],id_C=id_C[i])
        data_list.append(data)
    return data_list

### Model train
def run(device, train_dataset, valid_dataset, test_dataset, model, scheduler_name, loss_func, epochs=300, batch_size=32, 
        vt_batch_size=32, lr=0.001, lr_decay_factor=0.5, lr_decay_step_size=50, weight_decay=0, 
    save_dir='checkpoints/', disable_tqdm=False):     

#     model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters()) 
    print('num_parameters:', num_params)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, vt_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, vt_batch_size, shuffle=False)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if scheduler_name == 'steplr':
        scheduler = StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_factor)
    elif scheduler_name == 'onecyclelr':
        scheduler = OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(train_loader), epochs=epochs)
    else:
        scheduler = 0

    best_valid = float('inf')
    test_valid = float('inf')
        
    if save_dir != '':
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    start_epoch = 1
    
    train_mae_list = []
    valid_mae_list = []
    test_mae_list = []
    
    for epoch in range(start_epoch, epochs + 1):
        print("=====Epoch {}".format(epoch), flush=True)
        t_start = time.perf_counter()
        
        train_mae = train(model, optimizer, scheduler, scheduler_name, train_loader, loss_func, device, disable_tqdm)
        valid_mae = val(model, valid_loader, device, disable_tqdm)
        test_mae = val(model, test_loader, device, disable_tqdm)

        train_mae_list.append(train_mae)
        valid_mae_list.append(valid_mae)
        test_mae_list.append(test_mae)
        
        if valid_mae < best_valid:
            best_valid = valid_mae
            test_valid = test_mae
            if save_dir != '' and scheduler != 0:
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
                torch.save(checkpoint, os.path.join(save_dir, 'Best_valid_model_Matformer.pt'))
            elif save_dir != '' and scheduler == 0:
                print('Saving checkpoint...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_valid_mae': best_valid, 'num_params': num_params}
                torch.save(checkpoint, os.path.join(save_dir, 'Best_valid_model_Matformer.pt'))

        t_end = time.perf_counter()
        print({'Train': train_mae, 'Validation': valid_mae, 'Test': test_mae, 'Best valid': best_valid, 'Test@ best valid': test_valid, 'Duration': t_end-t_start})

        if scheduler_name == 'steplr':
            scheduler.step()
    
    # draw a train, valid, test plot vs epoch   
    def plot_metrics(train_mae_list, valid_mae_list, test_mae_list):
        plt.figure(figsize=(10, 6))
        plt.plot(train_mae_list, label='Train MAE')
        plt.plot(valid_mae_list, label='Validation MAE')
        plt.plot(test_mae_list, label='Test MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Training, Validation, and Test MAE vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.savefig('train_test_model_CGCNN.png')
        
    # Call the plot function after the training loop
    # plot_metrics(train_mae_list, valid_mae_list, test_mae_list)

    print(f'Best validation MAE so far: {best_valid}')
    print(f'Test MAE when got best validation result: {test_valid}')
    
        
def train(model, optimizer, scheduler, scheduler_name, train_loader, loss_func, device, disable_tqdm):  
    model.train()
    loss_accum = 0
    for step, batch_data in enumerate(tqdm(train_loader, disable=disable_tqdm)):
        optimizer.zero_grad()
        batch_data = batch_data.to(device)
        out = model(batch_data)
        loss = loss_func(out, batch_data.y.unsqueeze(1))
        loss.backward()
        optimizer.step()
        if scheduler_name == 'onecyclelr':
            scheduler.step()
        loss_accum += loss.detach().cpu().item()
    return loss_accum / (step + 1)

def val(model, data_loader, device, disable_tqdm):   
    model.eval()

    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    
    for step, batch_data in enumerate(tqdm(data_loader, disable=disable_tqdm)):
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        preds = torch.cat([preds, out.detach_()], dim=0)
        targets = torch.cat([targets, batch_data.y.unsqueeze(1)], dim=0)

    return torch.mean(torch.abs(preds - targets)).cpu().item()

parser = argparse.ArgumentParser(description='Fullerene')
parser.add_argument('--target', type=str, default='Eb')
parser.add_argument('--use_optimized_structure', default=False, action='store_true')
parser.add_argument('--train_ratio', type=float, default=0.8)
parser.add_argument('--valid_ratio', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--node_embedding', type=int, default=64)
parser.add_argument('--num_conv_layer', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--vt_batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=150)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--save_dir', type=str, default='checkpoints/')
parser.add_argument('--disable_tqdm', default=False, action='store_true')
parser.add_argument('--scheduler', type=str, default='onecyclelr')
parser.add_argument('--use_edge', default=False, action='store_true')
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--num_heads', type=int, default=4)


args = parser.parse_args()

# load data 
if args.use_optimized_structure:
    edge_index=torch.load("feature/opt_and_unopt_for_matformer/edge_index_matfomer_opt.pt")
    node_feature=torch.load( "feature/opt_and_unopt_for_matformer/node_feature_matfomer_opt.pt")
    edge_feature = torch.load('feature/opt_and_unopt_for_matformer/edge_feature_matfomer_opt.pt')
else:
    edge_index=torch.load("feature/opt_and_unopt_for_matformer/edge_index_matfomer_unopt.pt")
    node_feature=torch.load( "feature/opt_and_unopt_for_matformer/node_feature_matfomer_unopt.pt")
    edge_feature = torch.load("feature/opt_and_unopt_for_matformer/edge_feature_matfomer_unopt.pt")

edge_index_c60 = edge_index[:5770]
node_feature_c60 = node_feature[:5770]
edge_feature_c60 = edge_feature[:5770]

edge_index_c70 = edge_index[5770:5770+100]
node_feature_c70 = node_feature[5770:5770+100]
edge_feature_c70 = edge_feature[5770:5770+100]

edge_index_c72_100 = edge_index[5870:]
node_feature_c72_100 = node_feature[5870:]
edge_feature_c72_100 = edge_feature[5870:]

# load label from csv
df_c60 = pd.read_csv("data/c20-c60-dft-all.csv")
df_c70 = pd.read_csv("data/c70-100-isomers-Eb-Eg-logP.csv")
df_c72_100 = pd.read_csv("data/c62-c720-dft-all.csv")

data_list=load_data_list(node_feature_c60, edge_index_c60, edge_feature_c60, df_c60, tar=args.target)
test_dataset = data_list[3958:]

test_dataset_c70 = load_data_list(node_feature_c70, edge_index_c70, edge_feature_c70, df_c70, tar=args.target)
test_dataset_c72_100 = load_data_list(node_feature_c72_100, edge_index_c72_100, edge_feature_c72_100, df_c72_100, tar=args.target)

# shuffle data 
new_data_list=shuffle(data_list[:3958],random_state=args.seed)
# new_data_list=shuffle(data_list,random_state=args.seed)

# Calculate split indices
total_len = len(new_data_list)
train_len = int(0.8 * total_len)
# valid_len = int(0.1 * total_len)
# The remaining data will be used for testing

# Split the data
train_dataset = new_data_list[:train_len]
valid_dataset = new_data_list[train_len:]
# valid_dataset = new_data_list[train_len:train_len+valid_len]
# test_dataset = new_data_list[train_len+valid_len:]

model = Matformer(node_fea=args.node_embedding,edge_fea=19,atom_input_features=92,
              conv_layers=args.num_conv_layer, hidden_layer=args.hidden_channels, heads=args.num_heads)

loss_func = torch.nn.MSELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

run(device=device, 
    train_dataset=train_dataset, valid_dataset=valid_dataset, test_dataset=test_dataset, 
    model=model, scheduler_name=args.scheduler, loss_func=loss_func, 
    epochs=args.epochs, batch_size=args.batch_size, vt_batch_size=args.batch_size, 
    lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size, 
    weight_decay=args.weight_decay, save_dir=args.save_dir)

#### Testing #####
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error, root_mean_squared_error

def rmse(y_true, y_pred):  
    return root_mean_squared_error(y_true, y_pred)

def mse(y_ture,y_pred):
    return mean_squared_error(y_ture,y_pred)

def mae(y_ture,y_pred):
    return mean_absolute_error(y_ture,y_pred)

def get_predictions_and_targets(model, data_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    all_id = []
    
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            out = model(batch_data)
        all_preds.extend(out.cpu().numpy())
        all_targets.extend(batch_data.y.cpu().numpy())
        all_id.extend(batch_data.id_C)
        
    return all_preds, all_targets, all_id

def get_predictions_with_certain_test_set(model, train_loader, test_loader, device, train_name=None, test_name='test'):
    test_preds, test_targets, test_id = get_predictions_and_targets(model, test_loader, device)
    test_preds=[i[0] for i in test_preds] 
    print()
    if train_name:
        train_preds, train_targets, train_id = get_predictions_and_targets(model, train_loader, device)
        train_preds=[i[0] for i in train_preds]
        print(f'{train_name}_r2_score=',r2_score(train_targets, train_preds))
        print(f'{train_name}_rmse=',rmse(train_targets, train_preds))
        print(f'{train_name}_mse=',mse(train_targets, train_preds))
        print(f'{train_name}_mae=',mae(train_targets, train_preds))   

    print(f'{test_name}_test r2_score=',r2_score(test_targets,  test_preds))
    print(f'{test_name}_rmse=',rmse(test_targets,  test_preds))
    print(f'{test_name}_mse=',mse(test_targets,  test_preds))
    print(f'{test_name}_mae=',mae(test_targets,  test_preds))

checkpoint = torch.load('checkpoints/Best_valid_model_Matformer.pt')
model.load_state_dict(checkpoint['model_state_dict'])

train_loader = DataLoader(train_dataset, 32, shuffle=True)
test_loader_C60 = DataLoader(test_dataset, 32, shuffle=False)
test_loader_C70 = DataLoader(test_dataset_c70, 32, shuffle=False)
test_loader_C72_100 = DataLoader(test_dataset_c72_100, 32, shuffle=False)

get_predictions_with_certain_test_set(model, train_loader, test_loader_C60, device, train_name='train_C20_58', test_name='test_C60')
get_predictions_with_certain_test_set(model, train_loader, test_loader_C70, device, train_name=None, test_name='test_C70')
get_predictions_with_certain_test_set(model, train_loader, test_loader_C72_100, device, train_name=None, test_name='test_C72_100')