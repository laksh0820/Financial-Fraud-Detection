import torch
import warnings
import logging
import numpy as np
from train_utils import oversample_fraud_edges,stratified_temporal_split,FocalLoss
from utils import set_random_seeds
from data_loading import print_data_stats
from model import GATFraudDetector
from inference import evaluate_model
warnings.filterwarnings('ignore')

def train_single_experiment(seed, config, experiment_logger, data, device):
    """Train a single experiment with given seed and config"""
    
    # Set random seed
    set_random_seeds(seed)
    
    # Start experiment logging
    experiment_logger.start_experiment(seed, config)
    
    # Use stratified temporal split
    train_data, val_data, test_data = stratified_temporal_split(data)

    # Apply oversampling after splitting
    train_data_oversampled = oversample_fraud_edges(train_data, target_ratio=config['target_ratio'])
    print_data_stats(train_data_oversampled, f"Seed {seed} - After Oversampling Training Data")
    train_data = train_data_oversampled

    print_data_stats(train_data, f"Seed {seed} - Train Set")
    print_data_stats(val_data, f"Seed {seed} - Validation Set")
    print_data_stats(test_data, f"Seed {seed} - Test Set")

    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    # Initialize model
    model = GATFraudDetector(
        num_node_features=data.num_node_features,
        num_edge_features=data.edge_attr.size(-1),
        **config['model_params']
    ).to(device)
    
    # Calculate class weights
    train_labels = train_data.edge_label.cpu().numpy()
    class_counts = np.bincount(train_labels.astype(int))
    class_weights = len(train_labels) / (2 * class_counts)

    # Calculate alpha for focal loss (weight for fraud class)
    fraud_ratio = train_data.edge_label.mean().item()
    focal_alpha = 1 - fraud_ratio  # Higher alpha for minority class

    # Use multiple loss functions
    focal_loss = FocalLoss(alpha=focal_alpha, gamma=config['focal_gamma'])
    pos_weight = torch.tensor([(class_weights[1] / class_weights[0]) / (config['downsampled_upweight'])]).to(device)
    bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr=config['learning_rate'], 
                                weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)

    def train_epoch():
        model.train()
        optimizer.zero_grad()
        
        out = model(train_data)
        
        # Combine focal loss and BCE loss
        loss1 = focal_loss(out, train_data.edge_label.float())
        loss2 = bce_loss(out, train_data.edge_label.float())
        loss = config['focal_weight'] * loss1 + (1 - config['focal_weight']) * loss2
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        return loss.item()

    # Training loop
    best_val_f1 = 0  
    patience_counter = 0
    
    logging.info(f"\nStarting training for seed {seed}...")
    
    for epoch in range(config['max_epochs']):
        # Training
        train_loss = train_epoch()
        
        # Validation
        val_metrics = evaluate_model(model, val_data)
        
        # Log epoch results
        experiment_logger.log_epoch(epoch, train_loss, val_metrics)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['f1'])  
        
        # Early stopping
        if val_metrics['f1'] > best_val_f1:  
            best_val_f1 = val_metrics['f1']  
            patience_counter = 0
            # Save best model
            model_path = f'models/best_fraud_model_seed_{seed}.pth'
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        
        if epoch % 25 == 0:
            logging.info(f"Seed {seed} - Epoch {epoch:5d} | Loss: {train_loss:10.4f} | Val F1: {val_metrics['f1']:7.4f} | "  # Changed from Val AUC to Val F1
                  f"Val AUC: {val_metrics['auc']:6.4f} | Val AP: {val_metrics['ap']:6.4f}")  # Moved AUC to second position
        
        if patience_counter >= config['patience']:
            logging.info(f"Early stopping at epoch {epoch} for seed {seed}")
            break

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(f'models/best_fraud_model_seed_{seed}.pth'))
    test_metrics = evaluate_model(model, test_data)
    
    # End experiment logging
    experiment_logger.end_experiment(test_metrics, f'models/best_fraud_model_seed_{seed}.pth')
    
    return test_metrics, model

def run_seed_experiments(seeds, base_config, experiment_logger, data, device):
    """Run experiments with multiple random seeds"""
    
    logging.info(f"\nStarting experiments with {len(seeds)} different seeds...")
    logging.info(f"Seeds to test: {seeds}")
    
    results = []
    
    for i, seed in enumerate(seeds, 1):
        logging.info("\n" + "="*80)
        logging.info(f"EXPERIMENT {i}/{len(seeds)} - SEED: {seed}")
        
        try:
            test_metrics, model = train_single_experiment(seed, base_config, experiment_logger, data, device)
            results.append((seed, test_metrics))
            
            logging.info(f"Completed seed {seed}")
            logging.info("="*80)
            
        except Exception as e:
            logging.error(f"Error in experiment with seed {seed}: {str(e)}")
            continue
    
    return results
