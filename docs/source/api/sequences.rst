Sequences API
=============

Interaction Histories
---------------------

:class:`compresso_recsys.ItemSequences` is the chronological view of a split: for
each row, the catalog indices that row interacted with, in order, with duplicates
preserved. It is the counterpart to the ``csr_matrix`` views, holding exactly the
two things a sparse row cannot express.

A CSR row is a set. Sorting its ``indices`` discards order, and its ``data``
array holds one value per distinct column, so re-watching a film is a larger
number rather than a second event. Both are precisely what a sequential model
learns from, so a separate representation is needed rather than a convention on
top of the matrix.

What ``ItemSequences`` deliberately does *not* hold is padding, ``PAD``/``MASK``
tokens, a maximum length, or any truncation. Those are modelling decisions that
architectures disagree about — a GRU pads right, a causal transformer pads left,
and context windows differ — so they belong in a trainer rather than in a
checkpoint that every model reads. :class:`compresso_recsys.models.SequenceBatcher`
owns them.

Rows may be empty, and carry no identity of their own: row *i* of a sequence
addresses the same user as row *i* of the matrix beside it, exactly as the
existing ``csr_matrix`` sources are aligned. Buffers are read-only, so a
sequence handed to a trainer cannot be modified underneath it.

.. autoclass:: compresso_recsys.ItemSequences
   :members:

.. autofunction:: compresso_recsys.save_item_sequences

.. autofunction:: compresso_recsys.load_item_sequences

Which Split Modes Produce Them
------------------------------

Sequences require an ordering to preserve, so only the chronological split modes
build them. ``leave_last_out`` and ``temporal`` payloads carry
``x_train_sequences`` together with ``train_source_sequences``,
``val_source_sequences`` and ``test_source_sequences``; ``user_split`` and
``item_split`` carry none, and loading a checkpoint built by those modes — or one
built before sequences existed — yields ``None`` for each key. Every matrix model
still loads complete.

Each sequence addresses the same rows and columns as the matrix it accompanies,
per stage: a ``temporal`` split gives each window its own catalog, so the
relationship holds within a stage rather than across the checkpoint.

.. code-block:: python

   from compresso_recsys import load_recsys_split, read_checkpoint

   with read_checkpoint("ml1m-llo") as root:
       split = load_recsys_split(root)

   sequences = split["x_train_sequences"]
   sequences.n_rows              # rows, matching split["x_train"].shape[0]
   sequences.n_items             # catalog width, matching its column space
   sequences.row(0)              # one history, in order, repeats intact
   sequences.row_lengths         # events per row
   sequences.take_rows(0, 256)   # a contiguous batch
   sequences.select_rows([7, 2]) # an arbitrary selection, for shuffling
