"""Hardcoded, per-pattern explanations (used by TemplateExplainer).

Short analyst-style rationale for each known laundering pattern. Keys are the labels in
explanation_model.patterns.PATTERN_LABELS.
"""

PATTERN_EXPLANATIONS = {
    "fan-out": (
        "The account rapidly disperses funds to several distinct counterparties "
        "in a short time window. This fan-out structure is often used to break a "
        "large sum into smaller transfers and obscure its final destination."
    ),
    "fan-in": (
        "Several distinct accounts route funds into a single destination account "
        "within a short window. This fan-in structure is often used to consolidate "
        "funds from multiple sources before a final transfer out."
    ),
    "gather-scatter": (
        "Funds converge into an intermediary account from several sources and are "
        "then quickly redistributed onward. This gather-then-scatter sequence is "
        "characteristic of layering."
    ),
    "scatter-gather": (
        "Funds fan out from a single account to several intermediaries and are "
        "then reconsolidated into one account shortly after. This scatter-then-"
        "gather sequence obscures the funds' path through parallel branches."
    ),
    "simple cycle": (
        "Funds move through a short chain of accounts and return to (or near) the "
        "originating account. This closed loop has no obvious legitimate economic "
        "purpose and is a classic layering technique."
    ),
    "random": (
        "The transfers do not follow one clean canonical shape, but the irregular "
        "sequence of counterparties, payment formats, and short time gaps between "
        "hops is consistent with deliberate obfuscation rather than routine commerce."
    ),
    "bipartite": (
        "A dense many-to-many web of transfers connects two clusters of accounts, "
        "consistent with a mule network or a coordinated burst of transfers between "
        "two groups."
    ),
    "stack": (
        "Funds move through a linear chain of intermediary accounts in quick "
        "succession (A to B to C to ...), increasing the number of hops between "
        "origin and destination in a layering pattern."
    ),
}


UNKNOWN_EXPLANATION = (
    "The transaction was flagged as fraudulent by the prediction model, but its surrounding "
    "subgraph could not be matched to one of the known laundering patterns, so no pattern-specific "
    "explanation is available."
)
