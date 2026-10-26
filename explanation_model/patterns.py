"""Money-laundering pattern vocabulary shared by the classifier and the explainer."""

# names used in the BEGIN LAUNDERING ATTEMPT headers of the IBM AML *_Patterns.txt files
PATTERN_NAME_MAP = {
    "FAN-OUT": "fan-out",
    "FAN-IN": "fan-in",
    "GATHER-SCATTER": "gather-scatter",
    "SCATTER-GATHER": "scatter-gather",
    "CYCLE": "simple cycle",
    "RANDOM": "random",
    "BIPARTITE": "bipartite",
    "STACK": "stack",
}
PATTERN_LABELS = list(PATTERN_NAME_MAP.values())

# what a classifier returns when it cannot map its output to a known pattern
UNPARSED = "<unparsed>"
