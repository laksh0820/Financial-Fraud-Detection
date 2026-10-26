"""TemplateExplainer: fixed explanation text per laundering pattern."""
from typing import Optional

from explanation_model.explainer.base import Explanation, PatternExplainer
from explanation_model.explainer.templates import PATTERN_EXPLANATIONS, UNKNOWN_EXPLANATION
from explanation_model.subgraph import Subgraph


class TemplateExplainer(PatternExplainer):
    name = "template"

    def __init__(self, templates=None, unknown_text=UNKNOWN_EXPLANATION):
        self.templates = dict(PATTERN_EXPLANATIONS if templates is None else templates)
        self.unknown_text = unknown_text

    def explain(self, pattern: str, subgraph: Optional[Subgraph] = None) -> Explanation:
        text = self.templates.get(pattern)
        if text is None:
            return Explanation(pattern=pattern, text=self.unknown_text, details={"source": "template", "known_pattern": False})
        return Explanation(pattern=pattern, text=text, details={"source": "template", "known_pattern": True})
