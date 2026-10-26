import argparse

from common.config import DEFAULT_DATA_CONFIG, DEFAULT_MODEL_SETTINGS
from common.utils import set_seed, setup_logging as logger_setup  # re-exported: used all over prediction_model

__all__ = ["create_parser", "set_seed", "logger_setup"]


def create_parser():
    parser = argparse.ArgumentParser()

    #Config files
    parser.add_argument("--data_config", default=str(DEFAULT_DATA_CONFIG), help="Path to data_config.json (dataset / checkpoint folders)")
    parser.add_argument("--model_settings", default=str(DEFAULT_MODEL_SETTINGS), help="Path to model_settings.json (per-model default hyperparameters)")

    #Adaptations
    parser.add_argument("--emlps", action='store_true', help="Use emlps in GNN training")
    parser.add_argument("--reverse_mp", action='store_true', help="Use reverse MP in GNN training")
    parser.add_argument("--ports", action='store_true', help="Use port numberings in GNN training")
    parser.add_argument("--tds", action='store_true', help="Use time deltas (i.e. the time between subsequent transactions) in GNN training")
    parser.add_argument("--ego", action='store_true', help="Use ego IDs in GNN training")

    #Model parameters
    parser.add_argument("--batch_size", default=8192, type=int, help="Select the batch size for GNN training")
    parser.add_argument("--n_epochs", default=100, type=int, help="Select the number of epochs for GNN training")
    parser.add_argument('--num_neighs', nargs='+', type=int, default=[100,100], help='Pass the number of neighors to be sampled in each hop (descending).')

    #Misc
    parser.add_argument("--seed", default=1, type=int, help="Select the random seed for reproducability")
    parser.add_argument("--tqdm", action='store_true', help="Use tqdm logging (when running interactively in terminal)")
    parser.add_argument("--data", default=None, type=str, help="Select the AML dataset. Needs to be either small or medium.", required=True)
    parser.add_argument("--model", default=None, type=str, help="Select the model architecture. Needs to be one of [gin, gat, rgcn, pna]", required=True)
    parser.add_argument("--testing", action='store_true', help="Disable wandb logging while running the script in 'testing' mode.")
    parser.add_argument("--wandb_entity", default="jklagrawal123-iit-kharagpur", help="Weights & Biases entity (team / user)")
    parser.add_argument("--wandb_project", default="Fraud Detection System", help="Weights & Biases project")
    parser.add_argument("--save_model", action='store_true', help="Save the best model.")
    parser.add_argument("--unique_name", default=None, type=str, help="Unique name under which the model will be stored.")
    parser.add_argument("--finetune", action='store_true', help="Fine-tune a model. Note that args.unique_name needs to point to the pre-trained model.")
    parser.add_argument("--inference", action='store_true', help="Load a trained model and only do AML inference with it. args.unique name needs to point to the trained model.")
    parser.add_argument("--focal_loss", action='store_true', help="Use focal loss instead of cross entropy")
    parser.add_argument("--over_sample", action='store_true', help="Oversample fraud edges in the training data")
    parser.add_argument("--under_sample", action='store_true', help="Undersample non-fraud edges in the training data")
    parser.add_argument("--hybrid_sample", action='store_true', help="Undersample non-fraud edges then oversample fraud edges in the training data")
    parser.add_argument("--weighted_loader", action='store_true', help="Use WeightedLinkNeighborLoader for generating batches for training")
    # parser.add_argument('--class_weights', nargs=2, type=float, default=[0.5, 0.5], help='Class weights for sampling (weight_class_0 weight_class_1)')
    parser.add_argument('--alpha', type=float, default=0.75, help='Alpha parameter for focal loss')
    parser.add_argument('--gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
    return parser
