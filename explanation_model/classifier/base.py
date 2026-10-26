"""Interface every pattern classifier implements.

A classifier maps a serialized transaction `Subgraph` to one of the known money-laundering
patterns (explanation_model.patterns.PATTERN_LABELS). To add a new method (GNN-based, rule-based,
another LLM, ...) subclass `PatternClassifier` and register it in `classifier/__init__.py`.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from explanation_model.subgraph import Subgraph


@dataclass
class ClassificationResult:
    pattern: str                       # one of PATTERN_LABELS, or patterns.UNPARSED
    raw: Optional[str] = None          # raw model output, for debugging
    confidence: Optional[float] = None


class PatternClassifier(ABC):
    name = "base"

    @classmethod
    def add_cli_args(cls, parser):
        """Register the CLI flags this classifier needs (used by detect.py)."""

    @classmethod
    def from_args(cls, args) -> "PatternClassifier":
        """Build the classifier from parsed CLI args."""
        return cls()

    @abstractmethod
    def classify(self, subgraph: Subgraph) -> ClassificationResult:
        ...

    def classify_batch(self, subgraphs: List[Subgraph]) -> List[ClassificationResult]:
        return [self.classify(sg) for sg in subgraphs]
