import torch
import warnings
import json
import logging
from datetime import datetime
from data_loading import load_data, print_data_stats
from utils import setup_logging_and_dirs, log_system_info, set_random_seeds, ExperimentLogger, default_serializer
from train_utils import oversample_fraud_edges, stratified_temporal_split
from model import GATFraudDetector
from inference import plot_experiment_results, analyze_feature_importance
from training import run_seed_experiments
from comparison import compare_with_traditional_ml, plot_comparsion
warnings.filterwarnings('ignore')

# Initialize logging and experiment tracking
log_file = setup_logging_and_dirs()
log_system_info()
experiment_logger = ExperimentLogger()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

############################ BUILDING TRANSACTION GRAPH ############################

data = load_data('./dataset/transaction_graph.pth')
print_data_stats(data, "Original Dataset")

######################## COMPREHENSIVE EVALUATION ##################################

def main():
    """Main function to run all experiments"""
    
    # Configuration for experiments
    config = {
        'downsampled_upweight': 60,
        'target_ratio': 0.45,  # Oversampling target ratio
        'max_epochs': 500,
        'patience': 20,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'focal_gamma': 2.0,
        'focal_weight': 0.7,  # Weight for focal loss vs BCE loss
        'model_params': {
            'hidden_dim': 64,
            'heads': 4,
            'num_gnn_layers': 3,
            'dropout': 0.3
        }
    }
    
    # Random seeds to test
    seeds = [42, 123, 456, 789, 1337, 2023, 2024, 3141, 5432, 9876, 
             111, 222, 333, 444, 555, 666, 777, 888, 999, 1111]
    
    logging.info(f"\nConfiguration: {json.dumps(config, indent=2, default=default_serializer)}")
    
    # Run experiments
    results = run_seed_experiments(seeds, config, experiment_logger, data, device)
    
    # Print summary
    experiment_logger.print_summary()
    
    # Save results
    results_file = experiment_logger.save_results()
    
    # Create plots
    plot_experiment_results(experiment_logger)
    
    # Get best experiment and analyze it
    best_exp = experiment_logger.get_best_experiment()
    if best_exp:
        logging.info("\nANALYZING BEST MODEL")
        
        # Load best model for analysis
        best_seed = best_exp['seed']
        set_random_seeds(best_seed)
        
        # Recreate data splits with best seed
        train_data, val_data, test_data = stratified_temporal_split(data)
        train_data = oversample_fraud_edges(train_data, target_ratio=config['target_ratio'])
        test_data = test_data.to(device)
        
        # Load best model
        best_model = GATFraudDetector(
            num_node_features=data.num_node_features,
            num_edge_features=data.edge_attr.size(-1),
            **config['model_params']
        ).to(device)
        
        best_model.load_state_dict(torch.load(best_exp['model_path']))
        
        # Feature importance analysis
        feature_analysis = analyze_feature_importance(best_model, test_data)
        
        # Save feature analysis
        analysis_file = f"results/feature_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            json.dump({
                'best_experiment': best_exp,
                'feature_analysis': feature_analysis
            }, f, indent=2, default=default_serializer)
        logging.info(f"Saved feature analysis to {analysis_file}")
    
        comparison_results=compare_with_traditional_ml(train_data,val_data,test_data,best_exp['final_test_metrics'])
        plot_comparsion(comparison_results)
        
    
    logging.info("\nALL EXPERIMENTS COMPLETED!")
    logging.info(f"Results saved to: {results_file}")
    logging.info(f"Logs saved to: {log_file}")
    logging.info(f"Best model seed: {best_exp['seed'] if best_exp else 'N/A'}")
    logging.info(f"Best test F1: {best_exp['final_test_metrics']['f1']:.4f}" if best_exp else "N/A")  # Changed from AUC to F1

# Run all experiments
if __name__ == "__main__":
    main()