import os
import torch
import logging
import warnings
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
warnings.filterwarnings('ignore')

@torch.no_grad()
def evaluate_model(model, data, threshold=0.5):
    model.eval()
    out = model(data)
    probs = torch.sigmoid(out).cpu().numpy()
    labels = data.edge_label.cpu().numpy()
    
    # Calculate metrics
    auc = roc_auc_score(labels, probs)
    ap = average_precision_score(labels, probs)
    
    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    # Predictions with optimal threshold
    preds = (probs > optimal_threshold).astype(int)
    
    from sklearn.metrics import confusion_matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision_score * sensitivity) / (precision_score + sensitivity) if (precision_score + sensitivity) > 0 else 0
    
    return {
        'auc': auc,
        'ap': ap,
        'f1': f1,
        'precision': precision_score,
        'recall': sensitivity,
        'specificity': specificity,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': (tn, fp, fn, tp)
    }
    
def plot_experiment_results(experiment_logger, save_path='plots'):
    """Create comprehensive plots of experiment results"""
    
    if not experiment_logger.experiments:
        logging.warning("No experiments to plot")
        return
    
    # Create plots directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Plot 1: Test F1 vs Seed  # Changed comment from AUC to F1
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    seeds = [exp['seed'] for exp in experiment_logger.experiments]
    test_f1s = [exp['final_test_metrics']['f1'] for exp in experiment_logger.experiments]  # Changed from test_aucs
    test_aucs = [exp['final_test_metrics']['auc'] for exp in experiment_logger.experiments]  # Moved to second position
    val_f1s = [exp['best_val_f1'] for exp in experiment_logger.experiments]  # Changed from val_aucs
    training_times = [exp['training_time_seconds'] for exp in experiment_logger.experiments]
    
    # Test F1 by seed  # Changed from AUC to F1
    axes[0,0].scatter(seeds, test_f1s, alpha=0.7, s=60)  # Changed from test_aucs to test_f1s
    axes[0,0].set_xlabel('Random Seed')
    axes[0,0].set_ylabel('Test F1 Score')  # Changed from Test AUC
    axes[0,0].set_title('Test F1 Score by Random Seed')  # Changed from Test AUC
    axes[0,0].grid(True, alpha=0.3)
    
    # Test AUC by seed  # Changed from F1 to AUC
    axes[0,1].scatter(seeds, test_aucs, alpha=0.7, s=60, color='orange')  # Changed from test_f1s to test_aucs
    axes[0,1].set_xlabel('Random Seed')
    axes[0,1].set_ylabel('Test AUC')  # Changed from Test F1 Score
    axes[0,1].set_title('Test AUC by Random Seed')  # Changed from Test F1 Score
    axes[0,1].grid(True, alpha=0.3)
    
    # Validation vs Test F1  # Changed from AUC to F1
    axes[1,0].scatter(val_f1s, test_f1s, alpha=0.7, s=60, color='green')  # Changed variables
    axes[1,0].plot([min(val_f1s), max(val_f1s)], [min(val_f1s), max(val_f1s)], 'r--', alpha=0.5)  # Changed variables
    axes[1,0].set_xlabel('Validation F1')  # Changed from Validation AUC
    axes[1,0].set_ylabel('Test F1')  # Changed from Test AUC
    axes[1,0].set_title('Test vs Validation F1')  # Changed from AUC
    axes[1,0].grid(True, alpha=0.3)
    
    # Training time by seed
    axes[1,1].bar(range(len(seeds)), training_times, alpha=0.7, color='purple')
    axes[1,1].set_xlabel('Experiment Index')
    axes[1,1].set_ylabel('Training Time (seconds)')
    axes[1,1].set_title('Training Time by Experiment')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_path, f'seed_experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved experiment plots to {plot_path}")
    plt.show()
    
    # Plot 2: Training curves for best experiments
    best_experiments = sorted(experiment_logger.experiments, 
                            key=lambda x: x['final_test_metrics']['f1'],  # Changed from auc to f1
                            reverse=True)[:3]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green']
    for i, exp in enumerate(best_experiments):
        epochs = range(len(exp['train_losses']))
        val_f1s = [m['f1'] for m in exp['val_metrics']]  # Changed from val_aucs and auc to f1s and f1
        
        axes[0].plot(epochs, exp['train_losses'], color=colors[i], 
                    label=f"Seed {exp['seed']} (F1: {exp['final_test_metrics']['f1']:.4f})", alpha=0.8)
        axes[1].plot(epochs, val_f1s, color=colors[i], 
                    label=f"Seed {exp['seed']} (F1: {exp['final_test_metrics']['f1']:.4f})", alpha=0.8)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title('Training Loss - Best 3 Experiments')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation F1')  # Changed from Validation AUC
    axes[1].set_title('Validation F1 - Best 3 Experiments')  # Changed from AUC
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(save_path, f'training_curves_best_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(curves_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved training curves to {curves_path}")
    plt.show()

def analyze_feature_importance(model, test_data):
    """Analyze feature importance for the best model"""
    
    @torch.no_grad()
    def analyze_predictions():
        model.eval()
        
        # Get predictions for test set
        test_out = model(test_data).cpu().numpy()
        test_probs = 1 / (1 + np.exp(-test_out))  # Sigmoid
        
        # Analyze high-confidence fraud predictions
        fraud_threshold = 0.9
        high_conf_fraud = test_probs > fraud_threshold
        
        analysis_results = {}
        
        if high_conf_fraud.sum() > 0:
            logging.info(f"\nAnalysis of {high_conf_fraud.sum()} high-confidence fraud predictions:")
            
            # Analyze transaction types
            fraud_edge_types = test_data.edge_type[high_conf_fraud].cpu().numpy()
            type_names = ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"]
            
            type_distribution = {}
            logging.info("Transaction type distribution:")
            for i, name in enumerate(type_names):
                count = (fraud_edge_types == i).sum()
                if count > 0:
                    percentage = count/len(fraud_edge_types)*100
                    type_distribution[name] = {'count': int(count), 'percentage': percentage}
                    logging.info(f"  {name}: {count} ({percentage:.1f}%)")
            
            analysis_results['transaction_types'] = type_distribution
            
            # Analyze amounts
            fraud_amounts = test_data.edge_attr[high_conf_fraud, 0].cpu().numpy()
            amount_stats = {
                'mean': float(fraud_amounts.mean()),
                'median': float(np.median(fraud_amounts)),
                'std': float(fraud_amounts.std()),
                'min': float(fraud_amounts.min()),
                'max': float(fraud_amounts.max())
            }
            
            analysis_results['amount_statistics'] = amount_stats
            
            logging.info(f"\nAmount statistics:")
            logging.info(f"  Mean: ${amount_stats['mean']:,.2f}")
            logging.info(f"  Median: ${amount_stats['median']:,.2f}")
            logging.info(f"  Std: ${amount_stats['std']:,.2f}")
            logging.info(f"  Range: ${amount_stats['min']:,.2f} - ${amount_stats['max']:,.2f}")
        
        return analysis_results

    return analyze_predictions()

