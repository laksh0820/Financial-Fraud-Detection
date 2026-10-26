# AML Detection Framework

Money-laundering detection on the IBM AML transaction data, in two stages:

```
                      ┌───────────────────────── prediction_model ─────────────────────────┐
 transaction ───────► │  GNN (GIN / GAT / PNA / RGCN) predicts: fraud or not               │
                      └──────────────────────────────┬─────────────────────────────────────┘
                                     fraud?  ── no ──┴── yes ──┐   (explanation runs ONLY for predicted fraud)
                                                               ▼
                      ┌───────────────────────── explanation_model ────────────────────────┐
                      │ visualizer : GNNExplainer / CaptumExplainer edge importances + plot│
                      │ classifier : serialized subgraph  ──►  known laundering pattern    │
                      │ explainer  : pattern (+ subgraph) ──►  human-readable explanation  │
                      └────────────────────────────────────────────────────────────────────┘
```

## Layout

```
detect.py                         end-to-end CLI: predict -> (if fraud) classify + explain
configs/                          data_config.json (paths), model_settings.json (per-model hyper-parameters)
common/                           config loading, seeding / logging, jsonl helpers

prediction_model/                 everything GNN-specific
    main.py                       train / inference CLI
    predictor.py                  TransactionPredictor: load checkpoint, predict ONE transaction (used by detect.py)
    data_loading.py data_util.py  dataset, splits, sampling, PyG data objects
    models.py training.py train_util.py inference.py util.py
    preprocessing/format_kaggle_files.py   raw Kaggle csv -> formatted_transactions.csv (+ encoding maps)
    tuning/bayes_opt_focal.py, bayes_opt_focal_fig.py   Optuna search over focal-loss alpha / gamma

explanation_model/
    subgraph.py                   Subgraph / Edge: what classifier and explainer consume (+ text serialization)
    metadata.py                   EdgeMetadataStore: GNN ids -> real accounts / amounts / currencies / timestamps
    patterns.py                   the 8 laundering-pattern labels
    pipeline.py                   ExplanationPipeline: importance -> subgraph -> classifier -> explainer (-> plot)
    visualizer/                   importance.py (GNNExplainer/Captum), plotting.py, cases.py, cli.py
    classifier/                   base.py (interface), dataset.py (patterns file -> train/test samples)
        llm/                      prompts, LoRA fine-tuning, evaluation, LLMPatternClassifier, cli
    explainer/                    base.py (interface), template_explainer.py + templates.py (hardcoded text per pattern)
```

All commands are run from the repository root.

## Setup

```bash
python -m venv multignn --copies && source multignn/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install pyg-lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.8.0+cu126.html
pip install -r requirements.txt      # also pulls optuna / peft / bitsandbytes (LLM classifier, tuning)
```

Edit `configs/data_config.json` so `aml_data` / `model_to_load` / `model_to_save` point to your folders.

## 1. Prediction model (GNN)

```bash
# raw Kaggle csv -> <dataset>/formatted_transactions.csv + currency/payment_format/account maps + format_meta.json
python -m prediction_model.preprocessing.format_kaggle_files /path/to/raw_transactions.csv

# train (checkpoint_<unique_name>.tar is written to model_to_save)
python -m prediction_model.main --data Small_HI --model gin --focal_loss --seed 1 --alpha 0.75 --gamma 1.3 \
    --save_model --unique_name run1

# test-set F1 of a saved checkpoint
python -m prediction_model.main --data Small_HI --model gin --inference --unique_name run1

# focal-loss hyper-parameter search
python -m prediction_model.tuning.bayes_opt_focal --data Small_HI --model gin --n_trials 30 --n_epochs 20
```

The flags describing the architecture (`--model --emlps --reverse_mp --ports --tds --ego --num_neighs`) must be
identical when you later load the checkpoint for inference, visualization or `detect.py`.
`--wandb_entity` / `--wandb_project` configure Weights & Biases (`--testing` disables it).

## 2. Explanation model

**Visualizer** – importance plots for one transaction, or one TP / FP / FN / TN example:
```bash
python -m explanation_model.visualizer --data Small_HI --model gin --unique_name run1 [--explain_edge_idx 12345]
```

**Classifier** – the LLM learns to name the laundering pattern of a serialized subgraph. It is trained on the
laundering-attempt blocks of `*_Patterns.txt` (the whole block is the subgraph; blocks are split 80/20 per pattern so
no block is in both splits) and does not depend on the GNN:
```bash
python -m explanation_model.classifier.llm --patterns_file HI-Small_Patterns.txt --work_dir ./pattern_clf \
    --n_samples 2000 --base_model Qwen/Qwen3-14B --load_in_4bit
# zero-shot baseline on the same test split
python -m explanation_model.classifier.llm --patterns_file HI-Small_Patterns.txt --work_dir ./pattern_clf \
    --skip_build --skip_train --eval_zero_shot --load_in_4bit
```
Outputs in `--work_dir`: `train.jsonl`, `test.jsonl`, `adapter/`, `eval_results.csv`, `eval_metrics.json`.

**Explainer** – currently the fixed text per pattern in `explainer/templates.py`.

## 3. Detection: predict, and explain only what is flagged

```bash
python detect.py --data Small_HI --model gin --unique_name run1 --edge_ids 12345 67890 \
    --adapter_path ./pattern_clf/adapter --load_in_4bit --visualize --out_path detections.json

# or on random fraud transactions of the test set
python detect.py --data Small_HI --model gin --unique_name run1 --n_random_edges 5 --sample_label fraud \
    --adapter_path ./pattern_clf/adapter --load_in_4bit
```
For every transaction the JSON contains `gnn_pred`, `fraud_prob`, `actual_label`; **only if `gnn_pred == 1`** it also
contains an `explanation` block (`pattern`, `explanation`, the serialized `subgraph`, optional `plot_path`).
The LLM is loaded lazily, so a run that flags nothing never loads it.

## Extending

* **New classifier** (rule-based, GNN-based, another LLM, ...): subclass `PatternClassifier`
  (`classify(subgraph) -> ClassificationResult`), then `register_classifier("name", "pkg.module:Class")`
  (or add it to `_CLASSIFIERS` in `classifier/__init__.py`). It is selectable with `detect.py --classifier name`;
  `add_cli_args` / `from_args` let it bring its own flags.
* **New explainer** (e.g. one that names the concrete accounts / transactions forming the pattern): subclass
  `PatternExplainer` (`explain(pattern, subgraph) -> Explanation`) and register it the same way
  (`--explainer name`). It already receives the `Subgraph`, with `node_table()` and the edge list available.

## Notes

* **Prompt compatibility.** The LLM adapter was fine-tuned on the exact text in `classifier/llm/prompts.py` and
  `Subgraph.serialize()`. Changing either requires rebuilding the dataset and re-training.
* **Training vs. detection subgraphs.** The classifier is trained on complete laundering-attempt blocks. At detection
  time it sees the GNN-sampled neighborhood of the transaction, trimmed to the `--max_subgraph_edges` most important
  edges (`detect.py`), which can contain unrelated transactions. Check accuracy of that end-to-end path on known fraud
  edges before relying on it.
