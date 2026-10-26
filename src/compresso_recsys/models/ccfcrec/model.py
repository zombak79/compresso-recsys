"""The CCFCRec network: content and history encoders, and the two predictors."""

from __future__ import annotations

import torch
from torch import nn


class CCFCRec(nn.Module):
    """The encoders of Fig. 2.

    - ``content`` is ``g_c``, the two-layer LeakyReLU MLP of Eq. 1 that maps an
      item's content vector to its content-based embedding (CBCE) ``q_v``.
    - ``user_columns`` holds ``W_u`` of Eq. 3: a user's embedding (UCE) ``s_u``
      is the sum of the columns of the training items in their history.
    - ``item_columns`` holds ``W_v`` of Eq. 2: an item's co-occurrence
      embedding (COCE) ``z_v`` is the sum of the columns of its training users.

    ``cooccurrence`` caches ``z_v`` of every training item as of the last
    completed epoch, so prediction and a restored checkpoint need neither
    ``W_v`` nor the training interactions.
    """

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        n_features: int,
        embedding_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.content = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.user_columns = nn.Parameter(torch.empty(n_items, embedding_dim))
        self.item_columns = nn.Parameter(torch.empty(n_users, embedding_dim))
        self.register_buffer("cooccurrence", torch.zeros(n_items, embedding_dim))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        for layer in self.content:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.user_columns)
        nn.init.xavier_uniform_(self.item_columns)

    @staticmethod
    def contrastive_loss(
        content: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Eq. 4 for ``(B, d)`` CBCEs, ``(B, P, d)`` positive and ``(B, N, d)`` negative COCEs.

        Each of the ``P`` positives is contrasted with the same ``N``
        negatives, as the paper pairs 10 positives with 40 negatives into 400
        pairs. Scores are inner products over ``τ``.
        """
        positive = torch.einsum("bd,bpd->bp", content, positives) / temperature
        negative = torch.einsum("bd,bnd->bn", content, negatives) / temperature
        # -ln exp(p) / (exp(p) + Σ exp(n)) = logaddexp(p, lse(n)) - p
        others = torch.logsumexp(negative, dim=1, keepdim=True)
        return (torch.logaddexp(positive, others) - positive).mean()

    @staticmethod
    def bpr_loss(
        item: torch.Tensor,
        positive_users: torch.Tensor,
        negative_users: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. 7-8: ``-ln σ(⟨e_v, s_u+⟩ - ⟨e_v, s_u-⟩)``, averaged over the batch."""
        margin = (item * positive_users).sum(-1) - (item * negative_users).sum(-1)
        return nn.functional.softplus(-margin).mean()
