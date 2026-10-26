"""Explanation framework for transactions flagged as fraud by the prediction model.

    visualizer/   GNNExplainer / CaptumExplainer edge importances + plots for one transaction
    classifier/   serialized subgraph -> known money-laundering pattern (LLM today, pluggable)
    explainer/    detected pattern (+ subgraph) -> human-readable explanation (templates today, pluggable)

pipeline.py ties them together; subgraph.py / metadata.py / patterns.py are the shared data layer.
"""
