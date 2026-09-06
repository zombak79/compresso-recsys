Multimodal datasets and checkpoints
==========================================

The optional SWAP integration imports pretrained features from the versioned
`multimodal dataset release <https://zenodo.org/records/15403972>`_. ML-1M remains
the existing dataset; ``dbbook`` and ``lfm2k`` add book feedback and artist
listening counts. Cite both the underlying data and :ref:`cite-swap-multimodal`.

.. list-table:: Supported feature spaces
   :header-rows: 1

   * - Modality
     - Feature names
     - Datasets
   * - Text
     - ``text/minilm``, ``text/mpnet``
     - ML-1M, DBbook, Last.fm-2K
   * - Image
     - ``image/resnet152``, ``image/vgg``, ``image/vit_cls``, ``image/vit_avg``
     - ML-1M, DBbook, Last.fm-2K
   * - Audio
     - ``audio/vggish``, ``audio/whisper``
     - ML-1M, Last.fm-2K
   * - Video
     - ``video/i3d``, ``video/r2p1d``
     - ML-1M

Build or enrich
--------------------

Features are opt-in. Ordinary dataset builds do not download feature archives.
Choose feature spaces explicitly; importing a subset still downloads the full
dataset JSON archive once (approximately 264 MB ML-1M, 348 MB DBbook, 624 MB
Last.fm). Archives are cached under ``data/multimodal`` and checked against the
release's published checksums. Raw images/audio/video and encoders are not needed.

.. code-block:: python

   import compresso_recsys as cr

   checkpoint = cr.build_recsys_checkpoint(
       dataset="ml1m",
       checkpoint_path="artifacts/ml1m/multimodal.zip",
       min_entity_text_words=0,
       multimodal_features=["text/minilm", "image/resnet152"],
   )

   # Enrich an existing checkpoint without changing its split or item catalog.
   cr.enrich_multimodal_checkpoint(
       checkpoint,
       dataset="ml1m",
       features=["audio/vggish", "video/i3d"],
   )

.. code-block:: console

   compresso-recsys-build-checkpoint --dataset dbbook --split_mode item_split --multimodal_features text/minilm,image/resnet152 --checkpoint_path artifacts/dbbook/cold.zip
   compresso-recsys-build-checkpoint --dataset lfm2k --multimodal_features text/minilm,image/resnet152,audio/vggish --checkpoint_path artifacts/lfm2k/multimodal.zip

``enrich_multimodal_checkpoint`` also accepts ``archive_path`` for a local JSON
ZIP with the upstream filenames. Local imports record a SHA-256 but are marked
unverified; they are not claimed to be an official release. Imports reject
duplicate IDs, inconsistent dimensions, nonfinite vectors, unsupported feature
names, and zero catalog overlap. Checkpoint updates are atomic: a failed import
does not replace the original ZIP. Downloaded pickle files are never executed.

Reading features safely
---------------------------

.. code-block:: python

   with cr.read_checkpoint(checkpoint) as root:
       split = cr.load_recsys_split(root)
       spaces = cr.list_item_embeddings(root)
       features = cr.load_item_embeddings(
           root, "text/minilm", item_ids=split["train_item_ids"]
       )
       matrix = features["embeddings"]  # float32: requested items x dimensions
       available = features["available"]  # boolean: one entry per requested item

The optional ``item_ids`` argument reorders features by ID, not by assumed row
positions. Without it, the loader returns the stored catalog order. Unknown IDs
produce zero rows with ``available=False``. A valid all-zero vector is not
automatically considered missing. Feature coverage never filters interactions.
Choose an explicit policy for missing features before training or evaluating.

Each named feature space has its own matrix, item IDs, availability mask, and
manifest entry under ``item_embeddings``. Provenance includes the encoder filename,
source archive hash, normalization and pooling policy. Existing checkpoints need
no migration; listing their feature spaces returns an empty mapping.

For custom features, use ``save_item_embeddings(root, name, item_ids=...,
embeddings=..., available=..., metadata=...)`` inside ``update_checkpoint``.
The generic feature API lives in ``compresso_recsys.embeddings`` and is also
exported at package level. Loading a missing feature name raises ``KeyError``.

Dataset and evaluation details
----------------------------------

DBbook
~~~~~~~

``DBbook`` exposes original feedback and book titles/DBpedia mappings. The release
contains both zero and one ratings: builder defaults retain only ratings >= 1
before binarizing. Defaults otherwise use user support 5, item support 1,
500 validation users, 1,000 test users, and no minimum text length.

``DBbook.get_official_split()`` returns the supplied train and test DataFrames,
including negative feedback. The combined interaction table retains a
``source_split`` column. Random user/item split modes create new partitions.

For the supplied test boundary, use:

.. code-block:: console

   compresso-recsys-build-checkpoint --dataset dbbook --split_mode official --eval_draws 1 --checkpoint_path artifacts/dbbook/official.zip

This mode filters support using training data only, holds out validation edges
from supplied training interactions, and removes those edges from model training.
Test histories use the full supplied training history. The model is not refit
automatically. Test positives outside the training user/item vocabulary or below
evaluation support are excluded and counted in the manifest. Overlapping supplied
train/test pairs are rejected. It always uses one validation draw; ``val_users``
and ``test_users`` do not control this mode. This preserves the test boundary,
not an exact published training/evaluation recipe.

Last.fm-2K
~~~~~~~~~~~~~

``LastFM2K`` exposes listening counts and artist names/URLs. Defaults binarize
counts, require user support 5 and item support 1, and hold out 200 validation
and 400 test users. There is no minimum text length.

Media JSON keys can identify separate albums/songs for one artist. The importer
mean-pools by artist-ID prefix, following the authors' processing notebook, and
records this operation. It does not concatenate modalities or normalize vectors.
Tag-derived text is marked ``interaction_derived=True``: audit this information
before treating it as interaction-independent cold-start or historical metadata.

Neither DBbook ratings nor Last.fm listening counts provide interaction
timestamps. ``leave_last_out`` and ``temporal`` are rejected before downloading.
Last.fm tagging timestamps are not substituted for listening timestamps.

Comparison boundaries
~~~~~~~~~~~~~~~~~~~~~~~~

The authors' MMRec processing applies core-5 filtering, requires all modalities,
and remaps IDs. This integration uses original IDs and keeps incomplete modality
coverage. Its default checkpoints therefore do not reproduce their benchmark.
Coverage in the stored files may differ from headline counts; inspect each
manifest entry's ``available_items`` for your actual checkpoint.

The content recommender can consume feature matrices, but merely storing them
does not make collaborative-only models multimodal. Fit learned normalization,
dimensionality reduction, or fusion on training data only when evaluating cold
items. An item split alone does not establish that pretrained metadata is
historically available or free of interaction-derived information.

Usage terms
~~~~~~~~~~~~~~~

Review the upstream terms before using or redistributing these data. The supplied
Last.fm README specifies non-commercial use and refers commercial users to
Last.fm. ML-1M retains its original usage conditions. This integration does not
bundle the datasets or raw copyrighted media, or infer a blanket license from
the fact that pretrained features are downloadable.
