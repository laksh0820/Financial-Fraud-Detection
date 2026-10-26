"""Visualize one transaction: explainer importances -> plot."""
from explanation_model.visualizer.importance import compute_edge_importance
from explanation_model.visualizer.plotting import plot_explanation


def explain_and_plot(model, sample, edge_idx, args, case_key, case_label):
    """Computes GNNExplainer / CaptumExplainer importances for `sample` (from
    prediction_model.predictor.sample_and_predict) and saves the explanation plot. Returns its path."""
    is_hetero = bool(args.reverse_mp)
    importance = compute_edge_importance(model, sample, is_hetero)
    return plot_explanation(importance, sample, edge_idx, args, case_key, case_label, is_hetero)
