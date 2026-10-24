"""BERT4Rec: bidirectional self-attention trained with the Cloze objective.

Sun et al., *BERT4Rec: Sequential Recommendation with Bidirectional Encoder
Representations from Transformer*, CIKM 2019.

:class:`Bert4Rec` is the architecture of §3, :class:`Bert4RecConfig` its
published settings, and :class:`Bert4RecTrainer` the §3.6 objective plus the
batching the paper leaves to its implementation.
"""

from .config import Bert4RecConfig
from .model import Bert4Rec
from .trainer import Bert4RecTrainer

__all__ = ["Bert4Rec", "Bert4RecConfig", "Bert4RecTrainer"]
