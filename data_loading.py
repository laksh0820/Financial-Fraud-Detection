import torch
import os
import warnings
import logging
import pandas as pd
from torch_geometric.data import Data
warnings.filterwarnings('ignore')

def load_data(dataset_path):
    transaction_graph_path = dataset_path
    if os.path.exists(transaction_graph_path):
        # load the transaction graph
        data = torch.load(transaction_graph_path,weights_only=False)
        logging.info("Loaded existing transaction graph")
    else:
        logging.info("Building transaction graph from CSV...")
        # Load and preprocess data
        df = pd.read_csv("./transactions2.csv")
        df_fraud = df[df["isFraud"] == 1]
        df_non_fraud = df[df["isFraud"] == 0]
        df = pd.concat([df_fraud, df_non_fraud.iloc[:90000]],axis=0)
        
        # Assign unique IDs to all entities (customers + merchants + banks)
        all_entities = pd.concat([df["nameOrig"], df["nameDest"]]).unique()
        entity_to_id = {name: i for i, name in enumerate(all_entities)}
        
        # Classify nodes as Customer (C), Merchant (M), Banks(B)
        node_types = []
        for name in all_entities:
            if str(name).startswith("C"): 
                node_types.append(0)  # 0 = Customer
            elif str(name).startswith("M"):
                node_types.append(1)  # 1 = Merchant
            else:
                node_types.append(2)  # 2 = Bank
        
        node_types = torch.tensor(node_types, dtype=torch.long)
        
        # One-hot encode transaction types
        tx_type_mapping = {
            "TRANSFER": 0,
            "CASH_OUT": 1,
            "CASH_IN": 2,
            "PAYMENT": 3,
            "DEBIT": 4,
        }
        
        # Mapping each transaction type
        edge_types = torch.tensor(
            df["action"].map(tx_type_mapping).values,
            dtype=torch.long
        )
        
        # Node Features
        # Basic features: [node_type, avg_amount_sent, avg_amount_received, count_of_each_type_txs]
        node_features = []
        for name in all_entities:
            # For senders (nameOrig)    
            sent_txs = df[df["nameOrig"] == name]
            
            # For receivers (nameDest)
            received_txs = df[df["nameDest"] == name]
        
            # Calculate average amounts (handle empty cases)
            avg_amount_sent = sent_txs["amount"].mean() if not sent_txs.empty else 0.0
            avg_amount_received = received_txs["amount"].mean() if not received_txs.empty else 0.0
            
            # Calculate the number of each type of transaction
            txs  = pd.concat([sent_txs, received_txs])
            txs_count = { 
                'CASH_IN': len(txs[txs["action"] == "CASH_IN"]),
                'CASH_OUT': len(txs[txs["action"] == "CASH_OUT"]),
                'DEBIT': len(txs[txs["action"] == "DEBIT"]),
                'TRANSFER': len(txs[txs["action"] == "TRANSFER"]),
                'PAYMENT': len(txs[txs["action"] == "PAYMENT"])
            } 
        
            node_features.append([
                node_types[entity_to_id[name]].item(), # Convert tensor to scalar
                avg_amount_sent,
                avg_amount_received,
                txs_count['CASH_IN'],
                txs_count['CASH_OUT'],
                txs_count['DEBIT'],
                txs_count["TRANSFER"],
                txs_count['PAYMENT']
            ])
        
        node_features = torch.tensor(node_features, dtype=torch.float)
        
        # Edge Features
        # Edge indices: [2, num_edges] where each column is (sender, receiver)
        edge_index = torch.tensor([
            [entity_to_id[sender] for sender in df["nameOrig"]],     # Senders
            [entity_to_id[receiver] for receiver in df["nameDest"]]  # Receivers
        ], dtype=torch.long)
        
        # Edge attributes: [amount, step (time)]
        edge_attr = torch.tensor(
            df[["amount", "step"]].values,
            dtype=torch.float
        )
        
        # Assign fraud labels to edges (transactions)
        edge_label = torch.tensor(df["isFraud"].values, dtype=torch.float)
        
        # Create PyG Data object
        data = Data(
            x=node_features,          # Node features [num_nodes, num_features]
            edge_index=edge_index,    # Edge connections [2, num_edges]
            edge_attr=edge_attr,      # Edge features [num_edges, 2]
            edge_type=edge_types,     # Edge types [num_edges] (0=TRANSFER, etc.)
            edge_label=edge_label,    # Fraud labels [num_edges]
            num_nodes=len(all_entities)
        )

        # Save the transaction graph
        torch.save(data,'transaction_graph.pth')
        logging.info("Transaction graph built and saved")        
    return data

def print_data_stats(data, split_name="Dataset"):
    fraud_count = data.edge_label.sum().item()
    total_count = len(data.edge_label)
    non_fraud_count = total_count - fraud_count
    
    stats_msg = f"""
{split_name}: {data}
Number of nodes: {data.num_nodes}
Number of edges: {data.edge_index.size(1)}
Number of fraud transactions: {fraud_count}
Number of non-fraud transactions: {non_fraud_count}
Fraud ratio: {fraud_count / total_count:.4f}
Class imbalance ratio: {non_fraud_count / max(fraud_count, 1):.2f}:1"""
    logging.info(stats_msg)