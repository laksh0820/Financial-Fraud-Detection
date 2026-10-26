"""Interface every explainer implements.

An explainer turns the detected laundering pattern (and, optionally, the subgraph it was detected in)
into a human-readable explanation. Today: hardcoded text per pattern (TemplateExplainer). Later: e.g. an
explainer that names the concrete accounts / transactions that form the pattern -- it receives the
`Subgraph`, so nothing in the pipeline has to change. Register new ones in `explainer/__init__.py`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from explanation_model.subgraph import Subgraph


@dataclass
class Explanation:
    pattern: str
    text: str
    details: dict = field(default_factory=dict)   # explainer-specific extras (e.g. involved accounts)


class PatternExplainer(ABC):
    name = "base"

    @classmethod
    def add_cli_args(cls, parser):
        """Register the CLI flags this explainer needs (used by detect.py)."""

    @classmethod
    def from_args(cls, args) -> "PatternExplainer":
        return cls()

    @abstractmethod
    def explain(self, pattern: str, subgraph: Optional[Subgraph] = None) -> Explanation:
        ...
