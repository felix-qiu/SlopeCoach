"""Canonical JSON fingerprints for A9 semantic contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def semantic_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def semantic_snapshot(payload: Any) -> Any:
    """Return an isolated, strict-JSON-normalized semantic snapshot."""
    return json.loads(canonical_json(payload))
