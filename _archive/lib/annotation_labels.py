"""Parse movement annotations without silently turning malformed events into rest."""

from __future__ import annotations

import re

_MOVEMENT = re.compile(r"^ACT(?P<alias>\d*)_(?P<label>LEFT|RIGHT|DOWN)$", re.I)


def parse_act_label(description: str) -> str | None:
    """Normalize ACT/ACT<digits> movement labels; reject unknown ACT-like labels.

    The source corpus contains ACT1_right as well as ACT_right. Ignoring the
    numbered annotation incorrectly relabels an annotated movement as rest.
    Non-ACT descriptions (for example BAD_noise) are not movement labels.
    """
    value = str(description).strip()
    match = _MOVEMENT.fullmatch(value)
    if match is not None:
        return match.group("label").upper()
    if value.upper().startswith("ACT"):
        raise ValueError(f"Unrecognized ACT-like annotation: {description!r}")
    return None
