"""Explainers: detected laundering pattern (+ subgraph) -> human-readable explanation.

    from explanation_model.explainer import get_explainer_class
    explainer = get_explainer_class("template").from_args(args)
    explanation = explainer.explain("fan-out", subgraph)

Register another method with `register_explainer("name", "package.module:ClassName")`.
"""
import importlib

from explanation_model.explainer.base import Explanation, PatternExplainer

_EXPLAINERS = {
    "template": "explanation_model.explainer.template_explainer:TemplateExplainer",
}


def register_explainer(name, target):
    """`target` is a PatternExplainer subclass or a 'module.path:ClassName' string."""
    _EXPLAINERS[name] = target


def available_explainers():
    return sorted(_EXPLAINERS)


def get_explainer_class(name):
    if name not in _EXPLAINERS:
        raise KeyError(f"Unknown explainer '{name}'. Available: {available_explainers()}")
    target = _EXPLAINERS[name]
    if isinstance(target, str):
        module, cls_name = target.split(":")
        target = getattr(importlib.import_module(module), cls_name)
        _EXPLAINERS[name] = target
    return target
