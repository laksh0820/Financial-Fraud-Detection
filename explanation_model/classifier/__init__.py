"""Pattern classifiers: serialized subgraph -> known money-laundering pattern.

Registry of available implementations (lazy imports, so e.g. the LLM stack is only loaded when used):

    from explanation_model.classifier import get_classifier_class
    clf = get_classifier_class("llm").from_args(args)
    result = clf.classify(subgraph)

Register another method with `register_classifier("name", "package.module:ClassName")`.
"""
import importlib

from explanation_model.classifier.base import ClassificationResult, PatternClassifier

_CLASSIFIERS = {
    "llm": "explanation_model.classifier.llm.classifier:LLMPatternClassifier",
}


def register_classifier(name, target):
    """`target` is a PatternClassifier subclass or a 'module.path:ClassName' string."""
    _CLASSIFIERS[name] = target


def available_classifiers():
    return sorted(_CLASSIFIERS)


def get_classifier_class(name):
    if name not in _CLASSIFIERS:
        raise KeyError(f"Unknown classifier '{name}'. Available: {available_classifiers()}")
    target = _CLASSIFIERS[name]
    if isinstance(target, str):
        module, cls_name = target.split(":")
        target = getattr(importlib.import_module(module), cls_name)
        _CLASSIFIERS[name] = target
    return target
