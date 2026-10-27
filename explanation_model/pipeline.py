"""ExplanationPipeline: what happens to a transaction the prediction model flagged as fraud.

    importance (GNNExplainer / Captum) -> serialized subgraph -> classifier (pattern) -> explainer (text)
    plus an optional plot of the explained neighborhood (visualizer)

The classifier and explainer are injected, so any PatternClassifier / PatternExplainer implementation
can be used. The pipeline does NOT decide whether a transaction is fraud -- the caller (detect.py)
only invokes it when the GNN predicted fraud.
"""
import logging
from dataclasses import asdict

from explanation_model.classifier.base import PatternClassifier
from explanation_model.explainer.base import PatternExplainer
from explanation_model.metadata import EdgeMetadataStore
from explanation_model.visualizer.importance import compute_edge_importance
from explanation_model.visualizer.plotting import plot_explanation


class ExplanationPipeline:
    def __init__(self, classifier: PatternClassifier, explainer: PatternExplainer,
                 metadata: EdgeMetadataStore, max_subgraph_edges: int = 40, visualize: bool = False):
        self.classifier = classifier
        self.explainer = explainer
        self.metadata = metadata
        self.max_subgraph_edges = max_subgraph_edges
        self.visualize = visualize

    def run(self, model, sample, edge_idx, args) -> dict:
        """`sample` is the dict returned by prediction_model.predictor.sample_and_predict."""
        is_hetero = bool(args.reverse_mp)

        # 1. which edges of the sampled neighborhood mattered to the GNN's prediction
        importance = compute_edge_importance(model, sample, is_hetero)

        # 2. human-readable subgraph around the transaction (most important edges, chronological)
        subgraph = self.metadata.build_subgraph(
            sample['global_edge_ids'], edge_idx, importance.imp_by_id, self.max_subgraph_edges)

        # 3. which known laundering pattern does it look like?
        classification = self.classifier.classify(subgraph)

        # 4. why is that suspicious?
        explanation = self.explainer.explain(classification.pattern, subgraph)

        report = {
            "pattern": classification.pattern,
            "explanation": explanation.text,
            "explanation_details": explanation.details,
            "classifier_raw": classification.raw,
            "n_subgraph_edges": len(subgraph.edges),
            "subgraph": subgraph.serialize(),
        }
        if self.visualize:
            report["plot_path"] = plot_explanation(
                importance, sample, edge_idx, args, "PRED_FRAUD",
                f"Predicted fraud - pattern: {classification.pattern}", is_hetero,
                max_edges=self.max_subgraph_edges)  # same cap as the classifier's subgraph
        logging.info(f"[edge {edge_idx}] pattern = {classification.pattern}")
        return report
