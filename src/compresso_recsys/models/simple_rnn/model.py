"""The architecture: modules holding the learned parameters."""

from __future__ import annotations

from __future__ import annotations


import torch
from torch import nn



from compresso_recsys.models.simple_rnn.config import (
    RNNType,
)


class SimpleRNN(nn.Module):
    """Embedding, recurrence, and a linear head over the catalog.

    The head outputs ``n_items`` scores rather than ``vocab_size``: special
    tokens are never prediction targets, so giving them output columns would
    train weights that can only ever be wrong.

    :meth:`forward` returns states and :meth:`score` turns states into logits,
    kept separate because prediction needs logits at one position per row.
    Scoring first and gathering after would materialise
    ``rows x length x n_items``, which on a real catalog is where the memory
    goes.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        n_items: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        rnn_type: RNNType,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )
        recurrent = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = recurrent(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch applies this between layers only, so a single-layer RNN
            # would silently ignore it. self.dropout below covers both cases.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_items)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Hidden states for every position, shape ``(rows, length, hidden)``."""
        states, _ = self.rnn(self.embedding(tokens))
        return states

    def score(self, states: torch.Tensor) -> torch.Tensor:
        """Catalog logits for the given states, one score per item."""
        return self.head(self.dropout(states))
