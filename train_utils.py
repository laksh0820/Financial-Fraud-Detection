import torch
import warnings
import logging
import torch.nn.functional as F
from torch_geometric.data import Data
warnings.filterwarnings('ignore')

############################ SMOTE-like OVERSAMPLING #############################

def oversample_fraud_edges(data, target_ratio=0.3):
    """
    Oversample fraud edges by duplicating them with slight noise to balance classes
    """
    fraud_mask = data.edge_label == 1
    fraud_edges = data.edge_index[:, fraud_mask]
    fraud_attr = data.edge_attr[fraud_mask]
    fraud_labels = data.edge_label[fraud_mask]
    fraud_type = data.edge_type[fraud_mask]
    
    # Calculate how many fraud samples we need
    n_non_fraud = (data.edge_label == 0).sum().item()
    n_fraud_current = fraud_mask.sum().item()
    n_fraud_target = int(n_non_fraud * target_ratio / (1 - target_ratio))
    n_oversample = max(0, n_fraud_target - n_fraud_current)
    
    if n_oversample == 0:
        return data
    
    logging.info(f"Oversampling {n_oversample} fraud transactions...")
    
    # Randomly select fraud samples to duplicate
    oversample_indices = torch.randint(0, len(fraud_edges[0]), (n_oversample,))
    new_fraud_edges = fraud_edges[:, oversample_indices]
    new_fraud_attr = fraud_attr[oversample_indices]
    new_fraud_type = fraud_type[oversample_indices]
    
    # Add slight noise to avoid exact duplicates
    noise = torch.randn_like(new_fraud_attr) * 0.01 * new_fraud_attr.std()
    new_fraud_attr = new_fraud_attr + noise
    
    new_fraud_labels = torch.ones(n_oversample)
    
    # Combine with original data
    new_edge_index = torch.cat([data.edge_index, new_fraud_edges], dim=1)
    new_edge_attr = torch.cat([data.edge_attr, new_fraud_attr], dim=0)
    new_edge_label = torch.cat([data.edge_label, new_fraud_labels], dim=0)
    new_edge_type = torch.cat([data.edge_type, new_fraud_type],dim=0)
    
    return Data(
        x=data.x,
        edge_index=new_edge_index,
        edge_attr=new_edge_attr,
        edge_label=new_edge_label,
        edge_type=new_edge_type,
        num_nodes=data.num_nodes
    )

########################## STRATIFIED TEMPORAL SPLIT #############################

def stratified_temporal_split(data, time_feature_index=1, val_ratio=0.15, test_ratio=0.15):
    """
    Temporal split while maintaining class balance in each split
    """
    # Extract time steps from edge attributes
    time_steps = data.edge_attr[:, time_feature_index]
    
    # Get fraud labels
    fraud_labels = data.edge_label
    
    # Separate indices for fraud and non-fraud transactions
    fraud_indices = torch.where(fraud_labels == 1)[0]
    non_fraud_indices = torch.where(fraud_labels == 0)[0]
    
    # Sort both fraud and non-fraud indices by time
    fraud_time_sorted = fraud_indices[torch.argsort(time_steps[fraud_indices])]
    non_fraud_time_sorted = non_fraud_indices[torch.argsort(time_steps[non_fraud_indices])]
    
    # Calculate split sizes for each class
    n_fraud = len(fraud_time_sorted)
    n_non_fraud = len(non_fraud_time_sorted)
    
    n_fraud_test = int(n_fraud * test_ratio)
    n_fraud_val = int(n_fraud * val_ratio)
    n_fraud_train = n_fraud - n_fraud_val - n_fraud_test
    
    n_non_fraud_test = int(n_non_fraud * test_ratio)
    n_non_fraud_val = int(n_non_fraud * val_ratio)
    n_non_fraud_train = n_non_fraud - n_non_fraud_val - n_non_fraud_test
    
    # Create splits for fraud transactions
    fraud_train_indices = fraud_time_sorted[:n_fraud_train]
    fraud_val_indices = fraud_time_sorted[n_fraud_train:n_fraud_train + n_fraud_val]
    fraud_test_indices = fraud_time_sorted[n_fraud_train + n_fraud_val:]
    
    # Create splits for non-fraud transactions
    non_fraud_train_indices = non_fraud_time_sorted[:n_non_fraud_train]
    non_fraud_val_indices = non_fraud_time_sorted[n_non_fraud_train:n_non_fraud_train + n_non_fraud_val]
    non_fraud_test_indices = non_fraud_time_sorted[n_non_fraud_train + n_non_fraud_val:]
    
    # Combine indices for each split
    train_indices = torch.cat([fraud_train_indices, non_fraud_train_indices])
    val_indices = torch.cat([fraud_val_indices, non_fraud_val_indices])
    test_indices = torch.cat([fraud_test_indices, non_fraud_test_indices])
    
    # Create masks
    train_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
    val_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
    test_mask = torch.zeros(data.edge_index.size(1), dtype=torch.bool)
    
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    
    # Create split datasets
    train_data = Data(
        x=data.x,
        edge_index=data.edge_index[:, train_mask],
        edge_attr=data.edge_attr[train_mask],
        edge_label=data.edge_label[train_mask],
        edge_type=data.edge_type[train_mask],
        num_nodes=data.num_nodes
    )
    
    val_data = Data(
        x=data.x,
        edge_index=data.edge_index[:, val_mask],
        edge_attr=data.edge_attr[val_mask],
        edge_label=data.edge_label[val_mask],
        edge_type=data.edge_type[val_mask],
        num_nodes=data.num_nodes
    )
    
    test_data = Data(
        x=data.x,
        edge_index=data.edge_index[:, test_mask],
        edge_attr=data.edge_attr[test_mask],
        edge_label=data.edge_label[test_mask],
        edge_type=data.edge_type[test_mask],
        num_nodes=data.num_nodes
    )
    
    return train_data, val_data, test_data

######################### ADVANCED TRAINING WITH FOCAL LOSS #######################

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # weight for positive class (Fraud)
        self.gamma = gamma  # Focusing Parameter    
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1-pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
