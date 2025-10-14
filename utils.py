import torch
import os
import warnings
import json
import logging
import random
import numpy as np
from datetime import datetime
warnings.filterwarnings('ignore')

# Setup logging and experiment tracking
def setup_logging_and_dirs():
    """Setup logging directories and configuration"""
    # Create directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    # Setup logging
    log_filename = f"logs/fraud_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logging.info(f"Set random seed to: {seed}")

def log_system_info():
    """Log system and environment information"""
    logging.info("SYSTEM INFORMATION")
    logging.info(f"PyTorch Version: {torch.__version__}")
    logging.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"CUDA Device: {torch.cuda.get_device_name()}")
        logging.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logging.info(f"Number of CPU cores: {os.cpu_count()}")

def default_serializer(obj):
    if isinstance(obj, np.float32):
        return float(obj)  # Convert to a standard Python float
    if isinstance(obj, np.int64):
        return int(obj)    # Convert to a standard Python int
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

class ExperimentLogger:
    """Class to handle experiment logging and tracking"""
    
    def __init__(self, experiment_name="fraud_detection"):
        self.experiment_name = experiment_name
        self.experiments = []
        self.current_experiment = {}
        
    def start_experiment(self, seed, config):
        """Start a new experiment"""
        self.current_experiment = {
            'seed': seed,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'train_losses': [],
            'val_metrics': [],
            'best_val_f1': 0,  
            'best_epoch': 0,
            'final_test_metrics': {}
        }
        logging.info(f"Started experiment with seed {seed}")
        
    def log_epoch(self, epoch, train_loss, val_metrics):
        """Log epoch results"""
        self.current_experiment['train_losses'].append(train_loss)
        self.current_experiment['val_metrics'].append(val_metrics)
        
        if val_metrics['f1'] > self.current_experiment['best_val_f1']:  
            self.current_experiment['best_val_f1'] = val_metrics['f1'] 
            self.current_experiment['best_epoch'] = epoch
            
    def end_experiment(self, test_metrics, model_path):
        """End current experiment and save results"""
        self.current_experiment['end_time'] = datetime.now().isoformat()
        self.current_experiment['final_test_metrics'] = test_metrics
        self.current_experiment['model_path'] = model_path
        self.current_experiment['total_epochs'] = len(self.current_experiment['train_losses'])
        
        # Calculate training time
        start = datetime.fromisoformat(self.current_experiment['start_time'])
        end = datetime.fromisoformat(self.current_experiment['end_time'])
        self.current_experiment['training_time_seconds'] = (end - start).total_seconds()
        
        self.experiments.append(self.current_experiment.copy())
        
        # Log final results
        logging.info(f"Experiment completed - Seed: {self.current_experiment['seed']}")
        logging.info(f"\nTest AUC: {test_metrics['auc']:.4f}")
        logging.info(f"Test F1-Score: {test_metrics['f1']:.4f}")
        logging.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logging.info(f"Test Recall: {test_metrics['recall']:.4f}")
        logging.info(f"Test Specificity: {test_metrics['specificity']:.4f}")
        logging.info(f"Optimal Threshold: {test_metrics['optimal_threshold']:.4f}")
        tn, fp, fn, tp = test_metrics['confusion_matrix']
        logging.info(f"\nConfusion Matrix:")
        logging.info(f"True Negatives:  {tn}")
        logging.info(f"False Positives: {fp}")
        logging.info(f"False Negatives: {fn}")
        logging.info(f"True Positives:  {tp}")
        logging.info(f"\nTraining time: {self.current_experiment['training_time_seconds']:.1f} seconds")
        
    def save_results(self):
        """Save all experiment results to JSON"""
        results_file = f"results/experiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=default_serializer)
        logging.info(f"Saved experiment results to {results_file}")
        return results_file
        
    def get_best_experiment(self):
        """Get the experiment with best test F1-score"""  
        if not self.experiments:
            return None
        return max(self.experiments, key=lambda x: x['final_test_metrics']['f1'])  
        
    def print_summary(self):
        """Print summary of all experiments"""
        if not self.experiments:
            logging.info("No experiments completed yet.")
            return
            
        logging.info("\n" + "="*80)
        logging.info("EXPERIMENT SUMMARY")
        logging.info("="*80)
        
        # Sort experiments by test F1-score  
        sorted_experiments = sorted(self.experiments, key=lambda x: x['final_test_metrics']['f1'], reverse=True)  
        
        logging.info(f"{'Rank':<4} {'Seed':<8} {'Test F1':<9} {'Test AUC':<10} {'Val F1':<8} {'Epochs':<8} {'Time(s)':<8}")  
        logging.info("-" * 80)
        
        for i, exp in enumerate(sorted_experiments[:10], 1):  # Top 10
            test_f1 = exp['final_test_metrics']['f1'] 
            test_auc = exp['final_test_metrics']['auc']
            val_f1 = exp['best_val_f1']  
            epochs = exp['total_epochs']
            time_s = int(exp['training_time_seconds'])
            
            logging.info(f"{i:<4} {exp['seed']:<8} {test_f1:<9.4f} {test_auc:<10.4f} {val_f1:<8.4f} {epochs:<8} {time_s:<8}")  
        
        # Best experiment details
        best_exp = sorted_experiments[0]
        logging.info("\n" + "="*40)
        logging.info("BEST EXPERIMENT DETAILS")
        logging.info("="*40)
        logging.info(f"Seed: {best_exp['seed']}")
        logging.info(f"Test F1: {best_exp['final_test_metrics']['f1']:.4f}")  
        logging.info(f"Test AUC: {best_exp['final_test_metrics']['auc']:.4f}")
        logging.info(f"Test Precision: {best_exp['final_test_metrics']['precision']:.4f}")
        logging.info(f"Test Recall: {best_exp['final_test_metrics']['recall']:.4f}")
        logging.info(f"\nModel saved at: {best_exp['model_path']}")
