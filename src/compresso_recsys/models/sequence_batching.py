"""Public sequence batching, re-exported for custom training code.

The implementation lives in
:mod:`compresso_recsys.models.core.sequence_batching`; this module keeps the
documented import path stable.
"""

from .core.sequence_batching import SequenceBatcher

__all__ = [
    "SequenceBatcher",
]
