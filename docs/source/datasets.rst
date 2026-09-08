Datasets
========

Choose a dataset by its interactions, item metadata, and available features.
Multimodal data is not a separate dataset type: text, images, audio, and video
are metadata modalities. Some sources also supply **precomputed embeddings**;
otherwise, compute features with your own encoders and store them in the same
checkpoint format.

The table describes what the current adapters expose, not every resource that
might be available from the original dataset. Missing metadata and feature
coverage vary by item. See :ref:`cite-datasets` for citations,
:doc:`cli-reference` for all builder defaults, and :doc:`api/datasets` for classes.
Use :doc:`dataset-sweep` to collect comparable statistics and baseline scores.

.. list-table:: Dataset capabilities
   :header-rows: 1
   :widths: 18 34 30 18

   * - Dataset
     - Item metadata
     - Precomputed embeddings importable here
     - Interaction timestamps
   * - :ref:`dataset-ml1m`
     - Titles, genres, descriptions; optional features from plots/posters/trailers
     - Text, image, audio, video (SWAP)
     - Yes
   * - :ref:`dataset-ml20m`
     - Titles, genres, descriptions; optional tag annotations
     - None integrated
     - Yes
   * - :ref:`dataset-goodbooks`
     - Titles, authors, descriptions; optional tag annotations
     - None integrated
     - No
   * - :ref:`dataset-amazon2023`
     - Product text and optional image URLs
     - Compute text/image embeddings yourself
     - Yes
   * - :ref:`dataset-steam`
     - Titles, genres, tags, developers and other game fields
     - None integrated
     - Yes, day precision
   * - :ref:`dataset-netflix`
     - Movie titles and years
     - None integrated
     - Yes, day precision
   * - :ref:`dataset-taste-profile`
     - Item IDs only
     - None integrated
     - No
   * - :ref:`dataset-gowalla`
     - Location coordinates
     - None integrated
     - Yes
   * - :ref:`dataset-dbbook`
     - Book titles and DBpedia mappings; optional abstract/cover features
     - Text, image (SWAP)
     - No
   * - :ref:`dataset-lfm2k`
     - Artist names and page/image URLs; optional tag/cover/song features
     - Text, image, audio (SWAP)
     - No listening timestamps

The builder produces binary feedback by default after applying each dataset's
rating threshold and support filters. Raw adapters preserve source ratings or
counts. User/item splits and timestamp-based splits are different evaluation
protocols; metadata does not make an untimestamped dataset suitable for temporal
evaluation. See :doc:`dataset-validation` for paper comparison boundaries.

.. _dataset-ml1m:

MovieLens 1M
------------

Use ``dataset="ml1m"`` / ``MovieLens1M`` for timestamped movie ratings, titles,
genres, and additional beeFormer-generated descriptions. Builder defaults retain
ratings >= 4, require 5 interactions per user and 1 per item, and hold out 500
validation users and 1,000 test users. The default minimum text length is 30
words; use ``min_entity_text_words=0`` to avoid filtering on description length.

Optional SWAP embeddings provide text (plots), image (posters), audio, and video
(trailers). They enrich this existing dataset without changing its identity or
its split. Feature coverage does not filter the catalog.

.. code-block:: python

   import compresso_recsys as cr

   checkpoint = cr.build_recsys_checkpoint(
       dataset="ml1m",
       checkpoint_path="artifacts/ml1m/features.zip",
       min_entity_text_words=0,
       multimodal_features=["text/minilm", "image/resnet152"],
   )

   # Add more precomputed features without changing the existing split.
   cr.enrich_multimodal_checkpoint(
       checkpoint, dataset="ml1m", features=["audio/vggish", "video/i3d"]
   )

Sources: `MovieLens <https://grouplens.org/datasets/movielens/1m/>`_,
the beeFormer description source cited in :doc:`citing`, and the optional
`SWAP release <https://zenodo.org/records/15403972>`_. Ratings are cached in
``data/movielens1m/ml-1m.zip``; descriptions in
``data/movielens1m/item_text_descriptions.feather``. Original MovieLens usage
conditions still apply when adding features.

.. _dataset-ml20m:

MovieLens 20M
-------------

Use ``dataset="ml20m"`` / ``MovieLens20M`` for timestamped ratings, titles,
genres, and beeFormer-generated descriptions. ``annotation_source="ml20m_tags"``
adds user-tag annotations. No precomputed embedding import is integrated for
this dataset; compute features from its metadata when needed.

Builder defaults retain ratings >= 4, require user/item support 5/1, hold out
2,500 validation users and 5,000 test users, and require 30 text words.
These are not the MultVAE paper's held-out user counts; use the explicit recipe
in :doc:`dataset-validation` for that comparison.

Source: `MovieLens 20M <https://grouplens.org/datasets/movielens/20m/>`_.
Caches: ``data/movielens20m/ml-20m.zip`` and
``data/movielens20m/item_text_descriptions.feather``. Cite the description source
when using that additional metadata, and retain the original dataset terms.

.. _dataset-goodbooks:

Goodbooks-10k
-------------

Use ``dataset="goodbooks"`` / ``Goodbooks`` for book ratings, titles, authors,
average ratings, and beeFormer-generated descriptions. Optional
``annotation_source="goodbooks_tags"`` adds tag annotations. The adapter does
not expose book-cover URLs or precomputed embeddings; not every column from
the upstream source is retained in checkpoint metadata.

Builder defaults retain ratings >= 4, require user/item support 5/1, hold out
1,000 validation users and 2,500 test users, and require 30 text words. There
are no rating timestamps; use user/item splits, not temporal or LLO evaluation.

Source: `Goodbooks-10k <https://github.com/zygmuntz/goodbooks-10k>`_.
Caches: ``data/goodbooks/goodbooks-10k.zip`` and
``data/goodbooks/item_text_descriptions.feather``. Consult the source's terms
and cite both the dataset and any generated descriptions used.

.. _dataset-amazon2023:

Amazon Reviews 2023
--------------------

Use ``dataset="amazon2023"`` / ``AmazonReviews2023`` for category-specific,
timestamped product ratings and product metadata. The adapter downloads
rating-only interactions and item metadata, not review text. Product metadata
includes text such as titles, features, and descriptions where supplied.

**Amazon can be used for multimodal recommendation too.** Set
``include_image_urls=True`` to retain ``image_url`` and ``image_urls`` in item
metadata. These are links, not downloaded image files or image embeddings.
The user must obtain the referenced images as permitted by the source and
compute text/image embeddings with their own encoders. Store those features
using the same :ref:`dataset-item-embeddings` API as imported features.

.. code-block:: python

   import compresso_recsys as cr

   amazon_checkpoint = cr.build_recsys_checkpoint(
       dataset="amazon2023",
       amazon_category="Toys_and_Games",
       include_image_urls=True,
       min_entity_text_words=0,
       checkpoint_path="artifacts/amazon2023/toys.zip",
   )
   with cr.read_checkpoint(amazon_checkpoint) as root:
       metadata = cr.load_recsys_split(root)["entity_metadata"]
       # Feed entity_text and the image URLs to your own text/image pipeline.

Do not pass ``multimodal_features`` for Amazon: that convenience option imports
the SWAP release only. Generic ``save_item_embeddings`` works with any dataset.
Missing or unreachable image URLs need an explicit missing-feature policy.

Builder defaults use ``Toys_and_Games``, ratings >= 4, user/item support 20/20,
2,500 validation users, 5,000 test users, and a 30-word minimum text length.
See :doc:`cli-reference` for supported categories and cache/download options.
Source and terms: `Amazon Reviews 2023 <https://amazon-reviews-2023.github.io/>`_.

.. _dataset-steam:

Steam
-----

Use ``dataset="steam"`` / ``Steam`` for raw McAuley reviews and game metadata
with original product IDs. Every review is an implicit interaction, including
negative reviews. Metadata includes titles, genres, tags, developers, publishers,
release dates, prices, specifications, and URLs where supplied. This release
does not supply descriptions or integrated precomputed embeddings.

Defaults require user/item support 5/1, hold out 10,000 validation users and
10,000 test users, and impose no minimum text length. Missing metadata does not
remove interactions unless a positive ``min_entity_text_words`` is requested.

.. code-block:: python

   import compresso_recsys as cr

   checkpoint = cr.build_recsys_checkpoint(
       dataset="steam", split_mode="leave_last_out",
       checkpoint_path="artifacts/steam/llo.zip",
   )
   cold_checkpoint = cr.build_recsys_checkpoint(
       dataset="steam", split_mode="item_split",
       checkpoint_path="artifacts/steam/cold.zip",
       min_entity_text_words=1,
   )

Review dates have day precision; LLO ties retain source order. Use
``split_mode="temporal", temporal_period_hours=30*24`` for 30-day target windows;
the current protocol requires data spanning more than three such windows.
Compresso's LLO protocol has train/validation/test targets and is not an exact
BERT4Rec evaluation reproduction.

Source: `McAuley Steam data <https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data>`_.
Caches: ``data/steam/steam_reviews.json.gz`` and ``steam_games.json.gz``.
The first download is approximately 1.2 GB plus metadata. Canonical interactions
are parsed in 100,000-row batches and cached as Parquet; checkpoint construction
still needs memory proportional to the dataset. Retain upstream usage and
citation information.

.. _dataset-netflix:

Netflix Prize
-------------

Use ``dataset="netflix"`` / ``NetflixPrize`` for ratings with day-precision
dates and movie title/year metadata. There are no integrated precomputed
embeddings. Builder defaults retain ratings >= 4, require user/item support
5/1, hold out 40,000 validation and 40,000 test users, and impose no text-length
minimum. Defaults follow the MultVAE preprocessing convention, not an exact
reproduction of the authors' split.

Source: `Netflix Prize archive <https://archive.org/details/nf_prize_dataset.tar>`_.
Cache: ``data/netflix/nf_prize_dataset.tar.gz``. Nested ``training_set.tar`` and
archives containing per-movie files directly are supported. Original Netflix
usage terms apply; Compresso does not redistribute the data.

.. _dataset-taste-profile:

MSD Taste Profile
-----------------

Use ``dataset="taste-profile"`` / ``TasteProfile`` for user/song play counts.
The adapter exposes item IDs only: full Million Song Dataset metadata, mismatch
lists, and precomputed features are not included. Meaningful content-based
item cold-start experiments require an additional metadata source.

Builder defaults binarize counts, require 20 interactions per user and 200 per
item, hold out 50,000 validation and 50,000 test users, and impose no text-length
minimum. Play counts have no timestamps; LLO/temporal builds are rejected.
The preprocessing convention follows MultVAE, with Compresso's own splitting
and iterative support filtering.

Source: `MSD Taste Profile <http://millionsongdataset.com/tasteprofile/>`_.
Cache: ``data/taste-profile/train_triplets.txt.zip``. The configured original
archive uses HTTP; the old S3 mirror is unavailable. MSD/Echo Nest terms apply.

.. _dataset-gowalla:

Gowalla
-------

Use ``dataset="gowalla"`` / ``Gowalla`` for timestamped location check-ins and
latitude/longitude metadata. There is no item text, integrated precomputed
embedding set, or downloaded social graph. Coordinates can be inputs to a
user-defined item-feature pipeline.

Builder defaults require user/item support 10/10, hold out 10,000 validation
and 10,000 test users, and impose no text-length minimum. User/item CF splits
collapse repeated user-location pairs; ordered splits retain individual visits.
These support defaults follow NGCF, but this is the raw SNAP dataset, not the
authors' remapped LightGCN train/test files.

Source: `SNAP Gowalla <https://snap.stanford.edu/data/loc-gowalla.html>`_.
Cache: ``data/gowalla/loc-gowalla_totalCheckins.txt.gz``. Cite the source paper
and retain the source's usage information.

.. _dataset-dbbook:

DBbook
------

Use ``dataset="dbbook"`` / ``DBbook`` for book feedback, titles, and DBpedia
mappings. Optional precomputed text and image embeddings represent book
abstracts and covers. The adapter itself does not download raw cover images.

The interaction release contains both zero and one ratings: builder defaults
retain ratings >= 1 before binarizing. Other defaults are user/item support 5/1,
500 validation users, 1,000 test users, and no text-length minimum. There are
no rating timestamps; LLO/temporal builds are rejected before downloading.

.. code-block:: console

   compresso-recsys-build-checkpoint --dataset dbbook --split_mode item_split --multimodal_features text/minilm,image/resnet152 --checkpoint_path artifacts/dbbook/cold.zip

``DBbook.get_official_split()`` returns the supplied train and test DataFrames,
including negative feedback. The combined interaction table retains
``source_split``. Random user/item split modes create new partitions.

For the supplied test boundary, use:

.. code-block:: console

   compresso-recsys-build-checkpoint --dataset dbbook --split_mode official --eval_draws 1 --checkpoint_path artifacts/dbbook/official.zip

This mode filters support using training data only, withholds validation edges
from supplied training interactions, and removes those edges from model training.
Test histories use the full retained positive training history. The model is
not refit automatically. Test positives outside the training user/item vocabulary
or below evaluation support are excluded and counted in the manifest. Overlapping
supplied train/test pairs are rejected. It always uses one validation draw;
``val_users`` and ``test_users`` do not control this mode. This preserves the
test boundary, not an exact published training/evaluation recipe.

Source: the reconstructed DBbook data in the
`SWAP release <https://zenodo.org/records/15403972>`_. Cache:
``data/dbbook/dbbook_interaction_data.zip``. Consult the release's provenance
and usage information; this is not a download from the original ESWC service.

.. _dataset-lfm2k:

Last.fm-2K
----------

Use ``dataset="lfm2k"`` / ``LastFM2K`` for artist listening counts and names/URLs.
Optional precomputed text, image, and audio embeddings represent user tags,
album covers, and songs. Defaults binarize counts, require user/item support
5/1, hold out 200 validation users and 400 test users, and impose no text-length
minimum.

.. code-block:: console

   compresso-recsys-build-checkpoint --dataset lfm2k --multimodal_features text/minilm,image/resnet152,audio/vggish --checkpoint_path artifacts/lfm2k/features.zip

Media JSON keys can identify separate albums/songs for one artist. The importer
mean-pools by artist-ID prefix, following the authors' processing notebook, and
records this operation. It does not concatenate modalities or normalize vectors.
Tag-derived text is marked ``interaction_derived=True``: audit it before treating
it as interaction-independent cold-start or historical metadata.

There are no listening timestamps. LLO/temporal builds are rejected before
downloading; tagging timestamps are not substituted for listening timestamps.
Source: `HetRec 2011 <https://grouplens.org/datasets/hetrec-2011/>`_, via the
`SWAP release <https://zenodo.org/records/15403972>`_. Cache:
``data/lfm2k/lfm2k_interaction_data.zip``. The supplied README specifies
non-commercial use and refers commercial users to Last.fm. Cite Last.fm,
the suggested HetRec reference, and SWAP when using its enrichment.

.. _dataset-item-embeddings:

Using item embeddings
---------------------

The checkpoint feature format is the same for any dataset and modality.
Each named feature space has its own float32 matrix, item IDs, availability mask,
and manifest entry under ``item_embeddings``. Storing features does not make
collaborative-only models multimodal; a model must explicitly consume them.

Precomputed features
~~~~~~~~~~~~~~~~~~~~

The SWAP convenience importer supports these spaces. This is an inventory of
available embeddings, not a separate class of datasets.

.. list-table:: Precomputed feature names
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

Features are opt-in: ordinary builds do not download embedding archives. Selecting
a subset still downloads the full dataset JSON archive once (approximately
264 MB ML-1M, 348 MB DBbook, 624 MB Last.fm). They are cached under
``data/multimodal`` and checked against the release's published checksums.
Raw media and encoders are not needed for this import.

``enrich_multimodal_checkpoint`` also accepts ``archive_path`` for a local JSON
ZIP with the upstream filenames. Local imports record a SHA-256 but are marked
unverified. Imports reject duplicate IDs, inconsistent dimensions, nonfinite
vectors, unsupported feature names, and zero catalog overlap. Checkpoint updates
are atomic: a failed import does not replace the original ZIP. Downloaded pickle
files are never executed. Cite :ref:`cite-swap-multimodal` when using these features.

User-computed features
~~~~~~~~~~~~~~~~~~~~~~

For Amazon or any other dataset, run your chosen encoders outside the builder.
Keep a mapping from original item IDs to vectors for every feature space. Store
the available vectors; the loader can fill missing catalog items with a presence
mask. This helper accepts the output of your own text or image pipeline:

.. code-block:: python

   import numpy as np
   import compresso_recsys as cr

   def attach_computed_features(checkpoint_path, name, vectors_by_item_id, *, encoder):
       ids = np.asarray(list(vectors_by_item_id)).astype(str)
       vectors = np.asarray(list(vectors_by_item_id.values()), dtype=np.float32)
       with cr.update_checkpoint(checkpoint_path) as root:
           cr.save_item_embeddings(
               root, name, item_ids=ids, embeddings=vectors,
               metadata={"source": "user-computed", "encoder": encoder},
           )

Use names such as ``text/my_encoder`` and ``image/my_encoder``. Record encoder
versions, normalization, and data provenance in ``metadata`` for reproducibility.
This generic API does not fetch media or run encoders. Source usage terms still
apply to downloaded metadata and derived features.

Reading and evaluating features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   with cr.read_checkpoint(checkpoint) as root:
       split = cr.load_recsys_split(root)
       spaces = cr.list_item_embeddings(root)
       features = cr.load_item_embeddings(
           root, "text/minilm", item_ids=split["train_item_ids"]
       )
       matrix = features["embeddings"]
       available = features["available"]

``item_ids`` reorders rows by ID, not assumed positions. Without it, the loader
returns stored order. Unknown IDs produce zero rows with ``available=False``;
a valid zero vector is not automatically missing. Choose an explicit policy for
missing features before training/evaluation. Loading an absent feature name raises
``KeyError``. Old checkpoints need no migration; listing their spaces returns
an empty mapping. See :doc:`api/checkpoint` for the complete API.

Precomputed provenance includes encoder filenames, archive hashes, normalization,
and pooling. Coverage can differ from published headline counts: inspect each
manifest entry's ``available_items`` for the actual checkpoint. The SWAP MMRec
benchmark filters to all modalities and remaps IDs; these adapters preserve
original IDs and incomplete coverage, so defaults do not reproduce that benchmark.

Fit learned normalization, dimensionality reduction, and fusion on training data
only when evaluating cold items. An item split alone does not prove that metadata
was historically available or independent of held-out interactions.
