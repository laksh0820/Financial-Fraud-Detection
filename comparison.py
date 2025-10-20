import os
import logging
import numpy as np
import pandas as pd
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, f1_score,  average_precision_score, recall_score, precision_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

###################### COMPARISON WITH TRADITIONAL ML MODELS #######################
def prepare_tabular_features(data):
    """
    Convert graph data to tabular format for traditional ML models
    """
    # Extract edge features
    edge_features = data.edge_attr.cpu().numpy()
    
    # Extract node features for source and destination
    source_nodes = data.edge_index[0].cpu().numpy()
    dest_nodes = data.edge_index[1].cpu().numpy()
    
    source_features = data.x[source_nodes].cpu().numpy()
    dest_features = data.x[dest_nodes].cpu().numpy()
    
    # Combine all features
    tabular_features = np.concatenate([
        source_features, 
        dest_features, 
        edge_features
    ], axis=1)
    
    labels = data.edge_label.cpu().numpy()
    
    return tabular_features, labels

def compare_with_traditional_ml(train_data, val_data, test_data, gnn_test_metrics=None):
    """
    Compare GNN performance with traditional ML models
    """
    logging.info("\n" + "="*80)
    logging.info("COMPARISON WITH TRADITIONAL ML MODELS")
    logging.info("="*80)
    
    # Prepare tabular features
    X_train, y_train = prepare_tabular_features(train_data)
    X_val, y_val = prepare_tabular_features(val_data)
    X_test, y_test = prepare_tabular_features(test_data)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Define traditional models
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            random_state=42
        )
    }
    
    results = []
    
    # Train and evaluate each model
    for name, model in models.items():
        logging.info(f"\nTraining {name}...")
        
        try:
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Predict on test set
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_pred_proba)
            ap = average_precision_score(y_test, y_pred_proba)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            results.append({
                'Model': name,
                'AUC-ROC': auc,
                'AUC-PR': ap,
                'F1-Score': f1,
                'Precision': precision,
                'Recall': recall,
                'Specificity': specificity,
                'TP': tp,
                'FP': fp,
                'TN': tn,
                'FN': fn
            })
            
            logging.info(f"{name} - AUC: {auc:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            
        except Exception as e:
            logging.info(f"Error training {name}: {e}")
            continue
    
    # Add GNN results if provided
    if gnn_test_metrics:
        results.append({
            'Model': 'Best GAT Model',
            'AUC-ROC': gnn_test_metrics['auc'],
            'AUC-PR': gnn_test_metrics['ap'],
            'F1-Score': gnn_test_metrics['f1'],
            'Precision': gnn_test_metrics['precision'],
            'Recall': gnn_test_metrics['recall'],
            'Specificity': gnn_test_metrics['specificity'],
            'TP': gnn_test_metrics['confusion_matrix'][3],
            'FP': gnn_test_metrics['confusion_matrix'][1],
            'TN': gnn_test_metrics['confusion_matrix'][0],
            'FN': gnn_test_metrics['confusion_matrix'][2]
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Sort by F1-Score
    results_df = results_df.sort_values('F1-Score', ascending=False)
    
    # Display results
    logging.info("\n" + "="*80)
    logging.info("FINAL COMPARISON RESULTS (Sorted by F1-Score)")
    logging.info("="*80)
    
    display_columns = ['Model', 'F1-Score', 'AUC-ROC', 'Precision', 'Recall', 'Specificity']
    formatted_df = results_df[display_columns].copy()
    
    # Format numbers
    for col in display_columns[1:]:
        formatted_df[col] = formatted_df[col].apply(lambda x: f"{x:.4f}")
    
    logging.info(formatted_df.to_string(index=False))
    
    # Save results to CSV
    results_df.to_csv('models/model_comparison_results.csv', index=False)
    logging.info(f"\nComparison results saved to 'model_comparison_results.csv'")
    
    return results_df

def plot_comparsion(comparison_results,save_path='plots'):
    # Create comparison bar chart
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    metrics_to_plot = ['F1-Score', 'AUC-ROC', 'Precision', 'Recall']
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i//2, i%2]
        models = comparison_results['Model']
        values = comparison_results[metric]
        
        # Convert string values to float for plotting
        numeric_values = [float(v) if isinstance(v, str) else v for v in values]
        
        bars = ax.bar(models, numeric_values, color=colors[:len(models)], alpha=0.7)
        ax.set_title(f'{metric} Comparison')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, numeric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    model_comparison_path = os.path.join(save_path, f'model_comparison.png')
    plt.savefig(model_comparison_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved model comparison plot to {model_comparison_path}")
    plt.show()
