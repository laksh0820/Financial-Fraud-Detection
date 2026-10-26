"""Visualizer: GNNExplainer / CaptumExplainer importances and plots for a transaction under question.

    importance.py   compute_edge_importance(model, sample, is_hetero) -> EdgeImportance
    plotting.py     plot_explanation(...) -> PNG
    visualize.py    explain_and_plot(...) = importance + plot
    cases.py        find a TP / FP / FN / TN example on the test set
    cli.py          python -m explanation_model.visualizer ...
"""
