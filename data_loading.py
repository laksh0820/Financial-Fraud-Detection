import pandas as pd
import numpy as np
import torch
import logging
import itertools
from data_util import GraphData, HeteroData, z_norm, create_hetero_obj
from collections import defaultdict

def get_data(args, data_config):
    '''Loads the AML transaction data.
    
    1. The data is loaded from the csv and the necessary features are chosen.
    2. The data is split into training, validation and test data.
    3. PyG Data objects are created with the respective data splits.
    '''

    transaction_file = f"{data_config['paths']['aml_data']}/{args.data}/formatted_transactions.csv" #replace this with your path to the respective AML data objects
    df_edges = pd.read_csv(transaction_file)

    logging.info(f'Available Edge Features: {df_edges.columns.tolist()}')

    df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

    max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
    df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
    timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
    timestamps_int = timestamps.int()
    y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

    logging.info(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
    logging.info(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
    logging.info(f"Number of transactions = {df_edges.shape[0]}")

    edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
    node_features = ['Feature']

    logging.info(f'Edge features being used: {edge_features}')
    logging.info(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

    x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
    edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
    edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

    n_days = int(timestamps_int.max() / (3600 * 24) + 1)
    # n_days = int(timestamps.max() / (3600 * 24) + 1)
    n_samples = y.shape[0]
    logging.info(f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

    #data splitting
    daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], [] #irs = illicit ratios, inds = indices, trans = transactions
    for day in range(n_days):
        l = day * 24 * 3600
        r = (day + 1) * 24 * 3600
        day_inds = torch.where((timestamps_int >= l) & (timestamps_int < r))[0]
        # day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
        daily_irs.append(y[day_inds].float().mean())
        weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
        daily_inds.append(day_inds)
        daily_trans.append(day_inds.shape[0])
    
    split_per = [0.6, 0.2, 0.2]
    daily_totals = np.array(daily_trans)
    d_ts = daily_totals
    I = list(range(len(d_ts)))
    split_scores = dict()
    for i,j in itertools.combinations(I, 2):
        if j >= i:
            split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
            split_totals_sum = np.sum(split_totals)
            split_props = [v/split_totals_sum for v in split_totals]
            split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
            score = max(split_error) #- (split_totals_sum/total) + 1
            split_scores[(i,j)] = score
        else:
            continue

    i,j = min(split_scores, key=split_scores.get)
    #split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
    split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
    logging.info(f'Calculate split: {split}')

    #Now, we seperate the transactions based on their indices in the timestamp array
    split_inds = {k: [] for k in range(3)}
    for i in range(3):
        for day in split[i]:
            split_inds[i].append(daily_inds[day]) #split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately

    tr_inds = torch.cat(split_inds[0])
    val_inds = torch.cat(split_inds[1])
    te_inds = torch.cat(split_inds[2])    
    
    logging.info(f"Max training index: {tr_inds.max()}")
    logging.info(f"Training size: {len(tr_inds)}")   
    
    logging.info(f"Max validation index: {val_inds.max()}")
    logging.info(f"Validation size: {len(val_inds)}")  
    
    logging.info(f"Max test index: {te_inds.max()}")
    logging.info(f"Test size: {len(te_inds)}")     

    logging.info(f"Total train samples: {tr_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
            f"{y[tr_inds].float().mean() * 100 :.2f}% || Train days: {split[0][:5]}")
    logging.info(f"Total val samples: {val_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
        f"{y[val_inds].float().mean() * 100:.2f}% || Val days: {split[1][:5]}")
    logging.info(f"Total test samples: {te_inds.shape[0] / y.shape[0] * 100 :.2f}% || IR: "
        f"{y[te_inds].float().mean() * 100:.2f}% || Test days: {split[2][:5]}")

    # Program to calculate number of common accounts between fraud and non-fraud
    fraud_inds = torch.where(y == 1)[0]
    non_fraud_inds = torch.where(y == 0)[0]
    
    fraud_nodes = torch.unique(edge_index[:, fraud_inds])
    non_fraud_nodes = torch.unique(edge_index[:, non_fraud_inds])
    
    common_acc = torch.sum(torch.isin(fraud_nodes, non_fraud_nodes)).item()
    overlap_pct = common_acc / df_nodes.shape[0] * 100
    
    logging.info(f"Number of common accounts b/w fraud and non-fraud: {common_acc}")
    logging.info(f"Fraud-Nonfraud account overlap (out of total accounts): {overlap_pct:.1f}%")
    
    # Program to calculate number of common accounts between Training, Validation and Test split
    tr_nodes = torch.unique(edge_index[:, tr_inds])
    val_nodes = torch.unique(edge_index[:, val_inds])
    te_nodes = torch.unique(edge_index[:, te_inds])

    tr_val_overlap_pct = torch.sum(torch.isin(tr_nodes, val_nodes)).item() / len(val_nodes) * 100
    tr_te_overlap_pct = torch.sum(torch.isin(tr_nodes, te_nodes)).item() / len(te_nodes) * 100

    logging.info(f"Train-Val account overlap (out of total val accounts): {tr_val_overlap_pct:.1f}%")
    logging.info(f"Train-Test account overlap (out of total test accounts): {tr_te_overlap_pct:.1f}%")
    
    #Creating the final data objects
    tr_x, val_x, te_x = x, x, x
    # e_tr = tr_inds.numpy()
    e_val = np.concatenate([tr_inds, val_inds])
    
    if args.over_sample:
        tr_inds = over_sample(tr_inds, y, 100) # make the ratio fraud/non-fraud = 1%
    
    if args.under_sample:
        tr_inds = under_sample(tr_inds, y, 100, "random", edge_index[:, tr_inds])
        # tr_inds = under_sample(tr_inds, y, 100, "advance", edge_index[:, tr_inds])
        
    if args.hybrid_sample:
        # First under_sample then over_sample 
        tr_inds = under_sample(tr_inds, y, 100, "advance", edge_index[:, tr_inds])
        tr_inds = over_sample(tr_inds, y, 1)
    
    e_tr = tr_inds.numpy()
        
    tr_edge_index,  tr_edge_attr,  tr_y,  tr_edge_times  = edge_index[:,e_tr],  edge_attr[e_tr],  y[e_tr],  timestamps[e_tr]
    val_edge_index, val_edge_attr, val_y, val_edge_times = edge_index[:,e_val], edge_attr[e_val], y[e_val], timestamps[e_val]
    te_edge_index,  te_edge_attr,  te_y,  te_edge_times  = edge_index,          edge_attr,        y,        timestamps

    tr_data = GraphData (x=tr_x,  y=tr_y,  edge_index=tr_edge_index,  edge_attr=tr_edge_attr,  timestamps=tr_edge_times )
    val_data = GraphData(x=val_x, y=val_y, edge_index=val_edge_index, edge_attr=val_edge_attr, timestamps=val_edge_times)
    te_data = GraphData (x=te_x,  y=te_y,  edge_index=te_edge_index,  edge_attr=te_edge_attr,  timestamps=te_edge_times )
    
    #Adding ports and time-deltas if applicable
    if args.ports:
        logging.info(f"Start: adding ports")
        tr_data.add_ports()
        val_data.add_ports()
        te_data.add_ports()
        logging.info(f"Done: adding ports")
    if args.tds:
        logging.info(f"Start: adding time-deltas")
        tr_data.add_time_deltas()
        val_data.add_time_deltas()
        te_data.add_time_deltas()
        logging.info(f"Done: adding time-deltas")
    
    #Normalize data
    tr_data.x = val_data.x = te_data.x = z_norm(tr_data.x)
    if not args.model == 'rgcn':
        tr_data.edge_attr, val_data.edge_attr, te_data.edge_attr = z_norm(tr_data.edge_attr), z_norm(val_data.edge_attr), z_norm(te_data.edge_attr)
    else:
        tr_data.edge_attr[:, :-1], val_data.edge_attr[:, :-1], te_data.edge_attr[:, :-1] = z_norm(tr_data.edge_attr[:, :-1]), z_norm(val_data.edge_attr[:, :-1]), z_norm(te_data.edge_attr[:, :-1])

    #Create heterogenous if reverese MP is enabled
    #TODO: if I observe wierd behaviour, maybe add .detach.clone() to all torch tensors, but I don't think they're attached to any computation graph just yet
    if args.reverse_mp:
        tr_data = create_hetero_obj(tr_data.x,  tr_data.y,  tr_data.edge_index,  tr_data.edge_attr, tr_data.timestamps, args)
        val_data = create_hetero_obj(val_data.x,  val_data.y,  val_data.edge_index,  val_data.edge_attr, val_data.timestamps, args)
        te_data = create_hetero_obj(te_data.x,  te_data.y,  te_data.edge_index,  te_data.edge_attr, te_data.timestamps, args)
    
    logging.info(f'train data object: {tr_data}')
    logging.info(f'validation data object: {val_data}')
    logging.info(f'test data object: {te_data}')

    return tr_data, val_data, te_data, tr_inds, val_inds, te_inds

def over_sample(tr_inds, y, ratio):
    # Random Oversampling
    fraud_tr_inds = tr_inds[y[tr_inds] == 1]
    n_fraud = len(fraud_tr_inds)
    n_non_fraud = len(tr_inds) - len(fraud_tr_inds)
    extra_n_fraud = int(n_non_fraud // ratio) - n_fraud 
    if extra_n_fraud > 0:
        idx = torch.randint(0, len(fraud_tr_inds), (extra_n_fraud,))
        new_tr_inds = fraud_tr_inds[idx]
        tr_inds = torch.cat([tr_inds, new_tr_inds])
        tr_inds = torch.sort(tr_inds).values
    return tr_inds
    
def under_sample(tr_inds, y, ratio, method="random", edge_index=None):
    tr_inds = tr_inds.numpy()
    y_np = y.clone().numpy()
    edge_index_np = edge_index.clone().numpy()
    
    fraud_tr_inds = tr_inds[y_np[tr_inds] == 1]
    non_fraud_tr_inds = tr_inds[y_np[tr_inds] == 0]
    n_fraud = len(fraud_tr_inds)
    n_non_fraud = len(non_fraud_tr_inds)
    target_n_non_fraud = int(ratio * n_fraud)
    
    if method == "random":
        # Random Under-sampling
        if target_n_non_fraud < n_non_fraud:
            idx = np.random.permutation(len(non_fraud_tr_inds))[:target_n_non_fraud]
            new_non_fraud = non_fraud_tr_inds[idx]
            tr_inds = np.concatenate([fraud_tr_inds, new_non_fraud])
            tr_inds = np.sort(tr_inds)
        
    else:
        # Sophisticated Under-sampling
        tr_inds_to_idx = {}
        for idx, inds in enumerate(tr_inds):
            tr_inds_to_idx[inds] = idx
            
        edge_index_np = edge_index_np.T #converting from [2,N] to [N,2]
        fraud_mask = y_np[tr_inds] == 1
        fraud_edge_index = edge_index_np[fraud_mask]
        
        nbs = defaultdict(list)
        idx_to_tr_inds = {}
        for idx, (u, v) in enumerate(edge_index_np):
            nbs[u].append((v, fraud_mask[idx], idx))
            idx_to_tr_inds[idx] = tr_inds[idx]
        
        new_non_fraud_tr_inds = []

        nbs_edges = set()
        for u, v in fraud_edge_index:
            nbs_edges_u_idx = get_nbs_edges(nbs,u,hop=2) #get_nbs_edges: return non-fraud neighbour edges_index[0]
            nbs_edges_v_idx = get_nbs_edges(nbs,v,hop=2)
            for edge_idx in nbs_edges_u_idx:
                nbs_edges.add(idx_to_tr_inds[edge_idx])
            for edge_idx in nbs_edges_v_idx:
                nbs_edges.add(idx_to_tr_inds[edge_idx])
        
        while target_n_non_fraud > len(nbs_edges):      
            # Remove already selected non-frauds
            upd_non_fraud_tr_inds = []
            for inds in non_fraud_tr_inds:
                if inds not in nbs_edges:
                    upd_non_fraud_tr_inds.append(inds)
            non_fraud_tr_inds = np.array(upd_non_fraud_tr_inds)
        
            # Sample new non-frauds
            batch_size = min(len(non_fraud_tr_inds), args.batch_size)
            sampled_idx = np.random.permutation(len(non_fraud_tr_inds))[:batch_size]
            non_fraud_edge_index = []
            for idx in sampled_idx:
                inds = non_fraud_tr_inds[idx]
                edge_idx = tr_inds_to_idx[inds]
                non_fraud_edge_index.append(edge_index_np[edge_idx])
            for u, v in non_fraud_edge_index:
                nbs_edges_u_idx = get_nbs_edges(nbs,u,hop=2) #get_nbs_edges: return non-fraud neighbour edges_index[0]
                nbs_edges_v_idx = get_nbs_edges(nbs,v,hop=2)
                for edge_idx in nbs_edges_u_idx:
                    nbs_edges.add(idx_to_tr_inds[edge_idx])
                for edge_idx in nbs_edges_v_idx:
                    nbs_edges.add(idx_to_tr_inds[edge_idx])
        
        for inds in nbs_edges:
            new_non_fraud_tr_inds.append(inds)
        new_non_fraud_tr_inds = np.array(new_non_fraud_tr_inds)
        
        tr_inds = np.concatenate([fraud_tr_inds,  new_non_fraud_tr_inds])
        tr_inds = np.sort(tr_inds)
    
    return torch.from_numpy(tr_inds)

def get_nbs_edges(nbs, src, hop=2):
    edge_idx = set()
    visited = {src}
    current_level = [src]
    
    for current_hop in range(hop):
        next_level = []
        
        for node in current_level:
            for neighbor, is_fraud, edge_id in nbs.get(node, []):
                if is_fraud == 0:
                    edge_idx.add(edge_id)
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_level.append(neighbor)
        
        current_level = next_level
        if not current_level:
            break
    
    return list(edge_idx)
