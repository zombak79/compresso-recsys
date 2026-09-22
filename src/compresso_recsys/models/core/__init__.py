"""Machinery the models share, as opposed to the models themselves.

Nothing here implements a published method. These are the contracts, the
batching, the vocabularies and the validation that every recommender in the
package leans on, kept apart so that listing ``models/`` shows models.

The public surface is unchanged: names are re-exported from
``compresso_recsys.models``, and the documented module paths
(``models.base``, ``models.batching``, ``models.tokenizer``,
``models.sequence_batching``) still live one level up.
"""
