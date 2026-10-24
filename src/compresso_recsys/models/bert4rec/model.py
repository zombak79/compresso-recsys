"""The BERT4Rec architecture: §3 of Sun et al., CIKM 2019.

A stack of bidirectional transformer blocks over an item history, read out at
the positions holding ``[mask]``. The Cloze objective that trains it and the
batching that feeds it live in :mod:`.trainer`; this module is the network and
nothing else.

The output is *logits* over the vocabulary, not probabilities. Eq. 7 is written
as a softmax and Eq. 8 as the negative log-likelihood of it, but a softmax
followed by a log is the numerically poor way to evaluate that composition:
where a probability underflows to zero the log is infinite, and clamping it
makes the loss locally constant and its gradient exactly zero. Returning logits
lets :meth:`~.trainer.Bert4RecTrainer._train_step` use the fused
``cross_entropy``, which is the same function of the same inputs evaluated
stably. Ranking is unaffected either way, the softmax being monotonic.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["Bert4Rec", "PositionWiseFeedForward"]

# The reference's modeling.layer_norm calls tf.contrib.layers.layer_norm
# without an epsilon, which defaults to 1e-12; PyTorch's nn.LayerNorm
# defaults to 1e-5. SASRec's LAYER_NORM_EPS is 1e-8 for the same reason and
# a different reference: the number belongs to the implementation it was
# published with, so the two models disagreeing is correct.
LAYER_NORM_EPS = 1e-12

# Both GELU sites -- Eq. 3's feed-forward and the output head -- take PyTorch's
# default ``approximate='none'``, the exact ``x * Phi(x)`` §3.3 writes, Phi
# being the standard Gaussian cumulative distribution function. The
# reference's modeling.gelu is BERT's tanh approximation,
# ``0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))``, which differs from it by at
# most ~3e-4. The two disagree here, so the rule that settles LAYER_NORM_EPS
# does not apply: the reference supplies what the paper leaves unstated, not a
# second reading of what it states, and the paper states this one.


class PositionWiseFeedForward(nn.Module):
    """Eq. 3's two-layer feed-forward network, applied at every position.

    The inner width is ``4 * d_model``, which Eq. 3 fixes and the released
    config agrees with (``intermediate_size`` 256 at ``hidden_size`` 64), so it
    is not a parameter.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()

        self.linear1 = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(4 * d_model, d_model)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.gelu(self.linear1(states)))


class Bert4Rec(nn.Module):
    """Bidirectional transformer encoder scoring the catalog at ``[mask]``.

    Every position attends to every other, in both directions -- the property
    that separates this from a causal recommender like
    :class:`~compresso_recsys.models.SASRec`, and the reason the training
    objective has to mask: without a mask the target is in the input.

    Constructed by :meth:`~.trainer.Bert4RecTrainer._build_model`, which takes
    the architecture from :class:`~.config.Bert4RecConfig` and the token ids
    from the tokenizer, so the defaults here are the config's rather than a
    second opinion about them.
    """

    def __init__(
        self,
        n_items: int,
        max_history_length: int,
        d_model: int,
        n_blocks: int = 2,
        n_heads: int = 2,
        dropout: float = 0.2,
        att_dropout: float = 0.2,
        pad: int = 0,
        mask: int = 1,
        n_reserved: int = 3,
    ) -> None:
        super().__init__()

        self.n_items = n_items
        self.d_model = d_model
        self.max_history_length = max_history_length

        # The catalog sits above the reserved ids, so the table has a row for
        # every special as well as every item. The reference reaches the same
        # place from the other end: FreqVocab numbers items from 1 and appends
        # its three specials ([pad], [MASK], [NO_USE]) above them, leaving id 0
        # unassigned and therefore free for the zero-fill that pads a TFRecord.
        # Either way a padded position embeds to something that is not an item,
        # which is the property that matters. The third special differs -- unk
        # here against the reference's unused [NO_USE] -- so n_reserved is 3 in
        # both, but for one shared reason and one coincidence.
        self.n_reserved = n_reserved
        self.vocab_size = n_reserved + self.n_items

        self.item_embedding = nn.Embedding(
            self.vocab_size, self.d_model, padding_idx=pad
        )
        self.positional_embedding = nn.Embedding(self.max_history_length, self.d_model)
        # §3.4 writes h_i^0 as the plain sum of the item and position
        # embeddings -- in prose, not a numbered equation, so there is no Eq. to
        # cite here however much the rest of this file cites them -- but the
        # reference's embedding_postprocessor ends with
        # layer_norm_and_dropout(output, hidden_dropout_prob), so the sum is
        # normalized and then dropped out before the first block reads it.
        # Without the norm the first block attends over a state whose scale
        # is whatever the 0.02 initialization and the position table happen
        # to add up to; without the dropout, hidden_dropout_prob regularizes
        # two sites here against the reference's three.
        self.embedding_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        self.embedding_dropout = nn.Dropout(dropout)

        self.pad = pad
        self.mask = mask

        self.attention_layers = nn.ModuleList()
        self.attention_dropouts = nn.ModuleList()
        self.attention_norms = nn.ModuleList()

        self.feed_forward_layers = nn.ModuleList()
        self.feed_forward_dropouts = nn.ModuleList()
        self.feed_forward_norms = nn.ModuleList()

        for _ in range(n_blocks):
            attention_layer = nn.MultiheadAttention(
                d_model, n_heads, att_dropout, batch_first=True
            )
            attention_dropout = nn.Dropout(dropout)
            attention_normalization = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)

            feed_forward_layer = PositionWiseFeedForward(d_model)
            feed_forward_dropout = nn.Dropout(dropout)

            feed_forward_normalization = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)

            self.attention_layers.append(attention_layer)
            self.attention_dropouts.append(attention_dropout)
            self.attention_norms.append(attention_normalization)
            self.feed_forward_layers.append(feed_forward_layer)
            self.feed_forward_dropouts.append(feed_forward_dropout)
            self.feed_forward_norms.append(feed_forward_normalization)

        self.projection_layer = nn.Linear(d_model, d_model)
        self.gelu = nn.GELU()
        # Eq. 7 goes straight from the GELU to the tied embedding matrix, but
        # the reference's get_masked_lm_output normalizes in between
        # (dense -> gelu -> layer_norm -> matmul):
        # https://github.com/FeiSun/BERT4Rec/blob/b5a1c2eddbe5c2cb4ae6c7a7845b3e6f251f90ba/run.py#L366-L385
        # It is not cosmetic: the embeddings are initialized at a standard
        # deviation of 0.02, so without a normalized state the dot products
        # start within ~1e-3 of each other and the softmax is uniform to four
        # decimal places, which leaves almost no gradient for the blocks below.
        self.projection_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        self.output_bias = nn.Parameter(torch.zeros(self.vocab_size))

    def forward(
        self,
        item_history: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        mask_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score the vocabulary at every predicted position in the batch.

        Returns one row of logits per predicted position, in row-major order
        over ``item_history`` -- the order
        :meth:`~.trainer.Bert4RecTrainer._cloze_batch` builds its labels in, so
        the two line up.

        ``mask_positions`` marks which positions to read, and training has to
        pass it. Under the reference's ``mask_prob`` a chosen position becomes
        ``[mask]`` only some of the time; the rest keep the original item or
        take a random one, and Eq. 8 covers those too. Recovering the positions
        from ``item_history == mask`` would find fewer of them than there are
        labels, which is why the correspondence cannot rest on the token alone.
        Prediction passes nothing and falls back to it, the batch there being
        built with exactly one ``[mask]`` per row and no corruption at all.

        A batch with nothing to read would make that correspondence vacuous
        rather than wrong, and silently returning an empty score matrix hides
        the mistake downstream, so it is refused here.
        """
        item_history = item_history[..., -self.max_history_length :]

        embedded_items = self.item_embedding(item_history)

        positions = torch.arange(
            0, item_history.shape[-1], dtype=torch.long, device=item_history.device
        )
        embedded_positions = self.positional_embedding(positions)

        input_embedding = self.embedding_dropout(
            self.embedding_norm(embedded_items + embedded_positions)
        )

        trm_input_output = input_embedding

        # True marks a position attention must not read. Padding carries no
        # interaction, so letting a real position attend to it would mix a
        # filler embedding into the representation the Cloze target is read
        # from -- and the amount mixed in would depend on how much padding the
        # row happened to need, which is an artefact of batching rather than
        # anything about the user.
        #
        # PyTorch masks by adding -inf before the softmax, where the reference's
        # create_attention_mask_from_input_mask adds -10000.0 and so leaves
        # padding a weight of e^-10000 rather than zero. No row here is entirely
        # padding -- every window holds at least one interaction -- so the
        # softmax is well defined either way and the difference is unobservable.
        key_padding_mask = None
        if padding_mask is not None:
            key_padding_mask = ~padding_mask[..., -self.max_history_length :]

        for (
            attention_layer,
            attention_dropout,
            attention_normalization,
            feed_forward_layer,
            feed_forward_dropout,
            feed_forward_normalization,
        ) in zip(
            self.attention_layers,
            self.attention_dropouts,
            self.attention_norms,
            self.feed_forward_layers,
            self.feed_forward_dropouts,
            self.feed_forward_norms,
        ):
            applied_attention, _ = attention_layer(
                trm_input_output,
                trm_input_output,
                trm_input_output,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

            applied_attention_dropout = attention_dropout(applied_attention)

            in_between_residual = trm_input_output + applied_attention_dropout

            normalized_attention = attention_normalization(in_between_residual)

            applied_feed_forward = feed_forward_layer(normalized_attention)

            applied_feed_forward_dropout = feed_forward_dropout(applied_feed_forward)

            final_residual = normalized_attention + applied_feed_forward_dropout

            trm_input_output = feed_forward_normalization(final_residual)

        if mask_positions is None:
            selected = item_history == self.mask
        else:
            selected = mask_positions[..., -self.max_history_length :]
        masked_hidden = trm_input_output[torch.where(selected)]
        if masked_hidden.shape[0] == 0:
            raise ValueError(
                "Bert4Rec reads its output at the predicted positions and "
                "this batch has none: "
                + (
                    f"no token equals the mask id {self.mask}"
                    if mask_positions is None
                    else "mask_positions selects nothing"
                )
            )

        projection_linear = self.projection_layer(masked_hidden)
        projection_activation = self.gelu(projection_linear)
        projection_normalized = self.projection_norm(projection_activation)
        return projection_normalized @ self.item_embedding.weight.T + self.output_bias
