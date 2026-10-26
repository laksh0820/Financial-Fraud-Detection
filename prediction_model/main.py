"""Train / evaluate the GNN transaction classifier (the AML "prediction model").

    python -m prediction_model.main --data Small_HI --model gin --focal_loss --seed 1 \
        --alpha 0.75 --gamma 1.3 --save_model --unique_name run1
    python -m prediction_model.main --data Small_HI --model gin --inference --unique_name run1
"""
import logging
import time

from common.config import load_data_config
from prediction_model.util import create_parser, set_seed, logger_setup
from prediction_model.data_loading import get_data
from prediction_model.training import train_gnn
from prediction_model.inference import infer_gnn


def main():
    parser = create_parser()
    args = parser.parse_args()

    data_config = load_data_config(args.data_config)

    # Setup logging
    logger_setup()

    #set seed
    set_seed(args.seed)

    #get data
    logging.info("Retrieving data")
    t1 = time.perf_counter()

    tr_data, val_data, te_data, tr_inds, val_inds, te_inds = get_data(args, data_config)

    t2 = time.perf_counter()
    logging.info(f"Retrieved data in {t2-t1:.2f}s")

    if not args.inference:
        #Training
        logging.info(f"Running Training")
        train_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)
    else:
        #Inference
        logging.info(f"Running Inference")
        infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config)


if __name__ == "__main__":
    main()
