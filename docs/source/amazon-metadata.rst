Amazon text and image coverage
==============================

This audit analyzes cached item metadata for all 33 named Amazon subsets and
their four verified split profiles. Its curated text recipes are now installed
as per-subset preprocessing defaults for all four split strategies. The change
does not download images, create embeddings, or evaluate recommendation quality.
All **33 subsets / 132 profiles** completed the audit on 9 September 2026.
The :download:`machine-readable coverage archive <_static/amazon-metadata-coverage.json>`
contains the full counts, word-length quantiles, thresholds 1/5/10/20/30/50/100,
code fingerprints and source-cache file identities. Metadata cache sizes and
modification times match the original sealed-input manifests; retained catalogs
were rebuilt against the same checksum-verified inputs and frozen split code.

What the parameters actually do
-------------------------------

Amazon defaults to the category-specific ``metadata_text_fields`` listed under
:ref:`amazon-installed-text-recipes` and **min_entity_text_words=0**. The latter disables text-length
filtering; an item can have no description or even empty combined text and still
be included, provided its metadata record exists and it survives support pruning.

``metadata_text_fields`` selects top-level fields and nested dictionary paths
such as ``details.Brand`` in the requested order. Keys are case-sensitive;
missing or empty values are skipped. Adjacent nested selections share a single
``Details:`` label. JSON-encoded details and Parquet text arrays are supported.
An explicit field list replaces the category default; it is not appended to it.
Nonempty fields are joined into ``entity_text`` with field labels and blank lines.
A positive ``min_entity_text_words`` counts whitespace-separated words in that
**combined string, including labels**. It is not a description-presence test,
an encoder-token limit, or a semantic-quality measure. For example, a title and
categories can pass the threshold while ``description`` is completely absent.

Metadata filtering happens before support filtering and splitting. Removing
short-text items removes their interactions, potentially causing additional
users and items to fail support thresholds. The audit's threshold-loss figures
are **direct losses on the existing fixed catalogs/matrices**, not the sizes of
newly pruned or re-split checkpoints. For temporal, re-filtering can also change
timestamp endpoints and windows; it is not safe to reuse the old measured sizes.

Coverage definitions
--------------------

The category summary uses the union of retained item IDs across the four split
profiles, including the preprocessed catalogs for non-temporal profiles. It is
not the full raw Amazon catalog and does not weight popular items more heavily.
The split-level appendix separately measures the final catalog, observed training
items, validation targets and test targets. Training-pair loss is weighted by
unique nonzero user-item pairs in the original training matrix.

``Text`` means nonempty combined installed-default text after normalizing array containers;
``description`` and ``features`` mean nonempty content in those individual fields.
These are availability checks, not guarantees that the text is informative.
``Images`` means at least one recorded HTTP(S) **product-metadata image URL**.
URLs were not fetched: reachability, actual image contents, resolution, duplicates
and encoder success are unverified. Review text and customer-uploaded images are
not used. The upstream `field definitions
<https://amazon-reviews-2023.github.io/#data-fields>`_ distinguish product metadata
from review content.

Measured coverage by subset
---------------------------

The coverage and loss columns below use the **installed curated recipes**, as
measured in the original audit's ``curated`` result. The historical four-field
baseline is retained in the comparison tables and raw archive. Image URLs reflect
normalized source contents, not the adapter's current Parquet limitation.
The union counts may exceed a single split's size cap because the split catalogs
overlap only partially; no individual profile's size limits have changed.

.. list-table:: Installed-default metadata coverage and direct text-filter loss
   :header-rows: 1

   * - Subset
     - Union items
     - Text
     - Description
     - Features
     - Image URLs
     - Below 10 words
     - Below 30 words
   * - All_Beauty
     - 18,755
     - 100.00%
     - 15.71%
     - 18.36%
     - 100.00%
     - 1.21%
     - 40.47%
   * - Amazon_Fashion
     - 13,201
     - 100.00%
     - 10.86%
     - 61.76%
     - 100.00%
     - 0.82%
     - 72.44%
   * - Appliances
     - 22,334
     - 100.00%
     - 57.35%
     - 90.32%
     - 100.00%
     - 0.05%
     - 3.56%
   * - Arts_Crafts_and_Sewing
     - 19,690
     - 100.00%
     - 55.59%
     - 94.95%
     - 99.95%
     - 0.01%
     - 0.69%
   * - Automotive
     - 20,809
     - 100.00%
     - 62.91%
     - 97.54%
     - >99.99%
     - <0.01%
     - 0.45%
   * - Baby_Products
     - 20,108
     - 100.00%
     - 57.30%
     - 94.86%
     - 99.99%
     - 0.02%
     - 1.36%
   * - Beauty_and_Personal_Care
     - 19,153
     - 100.00%
     - 49.47%
     - 90.95%
     - 100.00%
     - 0.00%
     - 0.72%
   * - Books
     - 127,961
     - 100.00%
     - 77.61%
     - 98.76%
     - 88.66%
     - 0.04%
     - 0.39%
   * - CDs_and_Vinyl
     - 23,072
     - 100.00%
     - 90.69%
     - 0.59%
     - 99.92%
     - 0.00%
     - 8.43%
   * - Cell_Phones_and_Accessories
     - 23,458
     - 100.00%
     - 34.99%
     - 91.99%
     - 100.00%
     - 0.02%
     - 0.74%
   * - Clothing_Shoes_and_Jewelry
     - 24,842
     - 100.00%
     - 38.57%
     - 99.56%
     - 99.99%
     - 0.00%
     - 0.14%
   * - Digital_Music
     - 18,995
     - 100.00%
     - 42.30%
     - 0.09%
     - 100.00%
     - 10.73%
     - 63.81%
   * - Electronics
     - 99,261
     - 100.00%
     - 55.74%
     - 92.28%
     - >99.99%
     - 0.02%
     - 1.16%
   * - Gift_Cards
     - 749
     - 100.00%
     - 77.57%
     - 91.72%
     - 100.00%
     - 0.13%
     - 3.74%
   * - Grocery_and_Gourmet_Food
     - 21,273
     - 100.00%
     - 70.13%
     - 94.01%
     - 99.95%
     - 0.03%
     - 1.30%
   * - Handmade_Products
     - 23,704
     - 100.00%
     - 81.05%
     - 56.46%
     - 100.00%
     - <0.01%
     - 2.13%
   * - Health_and_Household
     - 23,856
     - 100.00%
     - 53.95%
     - 96.78%
     - 100.00%
     - 0.00%
     - 0.16%
   * - Health_and_Personal_Care
     - 17,288
     - 100.00%
     - 28.08%
     - 28.31%
     - 100.00%
     - 2.54%
     - 45.14%
   * - Home_and_Kitchen
     - 18,990
     - 100.00%
     - 60.65%
     - 98.35%
     - 100.00%
     - 0.00%
     - 0.18%
   * - Industrial_and_Scientific
     - 22,281
     - 100.00%
     - 54.32%
     - 93.18%
     - 100.00%
     - 0.01%
     - 0.86%
   * - Kindle_Store
     - 23,549
     - 100.00%
     - 68.42%
     - 99.96%
     - 89.32%
     - 0.00%
     - <0.01%
   * - Magazine_Subscriptions
     - 2,003
     - 100.00%
     - 42.69%
     - 0.00%
     - 100.00%
     - 6.64%
     - 57.96%
   * - Movies_and_TV
     - 24,791
     - 100.00%
     - 51.55%
     - 6.03%
     - 100.00%
     - 0.63%
     - 33.02%
   * - Musical_Instruments
     - 20,065
     - 100.00%
     - 67.55%
     - 95.38%
     - 99.98%
     - 0.04%
     - 1.49%
   * - Office_Products
     - 22,096
     - 100.00%
     - 56.08%
     - 96.37%
     - 99.70%
     - <0.01%
     - 0.33%
   * - Patio_Lawn_and_Garden
     - 17,357
     - 100.00%
     - 57.25%
     - 96.79%
     - 99.97%
     - 0.00%
     - 0.14%
   * - Pet_Supplies
     - 22,828
     - 100.00%
     - 59.58%
     - 96.82%
     - 99.98%
     - 0.01%
     - 0.49%
   * - Software
     - 18,322
     - 100.00%
     - 98.49%
     - 99.25%
     - >99.99%
     - 0.02%
     - 0.88%
   * - Sports_and_Outdoors
     - 25,641
     - 100.00%
     - 50.72%
     - 97.30%
     - 100.00%
     - 0.02%
     - 0.40%
   * - Subscription_Boxes
     - 327
     - 99.69%
     - 0.00%
     - 96.33%
     - 100.00%
     - 0.31%
     - 7.03%
   * - Tools_and_Home_Improvement
     - 19,936
     - 100.00%
     - 62.86%
     - 97.73%
     - >99.99%
     - 0.00%
     - 0.21%
   * - Toys_and_Games
     - 23,646
     - 100.00%
     - 67.03%
     - 98.88%
     - >99.99%
     - <0.01%
     - 0.18%
   * - Video_Games
     - 23,083
     - 100.00%
     - 69.08%
     - 89.16%
     - >99.99%
     - 0.11%
     - 2.80%

Source coverage versus current loader behavior
----------------------------------------------

The cached Hugging Face Parquet categories encode image records as a dictionary
of array columns. The current adapter expects a list of dictionaries with scalar
URL strings, so it fails to expose these URLs through ``include_image_urls=True``.
This is a loader limitation, not absent images. Parquet text-list cells also arrive
as NumPy arrays. **Text-array handling is now fixed**: the formatter joins their
contents and skips empty arrays, just as it does for JSON lists. It previously
stringified arrays, including brackets and empty ``[]`` values.

The table below preserves the audit's **pre-change** word counts and normalized
four-field comparison. The installed defaults additionally include curated
attributes; their results appear in the other tables. The image-loader issues
remain unresolved and need correction before image embedding builds. Positive
text thresholds require a fresh split verification; the default remains zero.

Prime Video records inside **Movies and TV** add a second format: their image
keys use width names such as ``720w`` and ``1920w``, which the current extractor
also ignores. Source coverage in this report includes all recorded HTTP(S) image
variants, including those widths; the preliminary 47.63% estimate based only on
``hi_res/large/thumb`` was incomplete and is superseded by this corrected audit.

.. list-table:: Historical four-field baseline: source versus pre-change adapter
   :header-rows: 1

   * - Subset
     - Source image URLs
     - Adapter image URLs
     - Text affected by old array formatting
     - Below 30: old native
     - Below 30: normalized four fields
   * - All_Beauty
     - 100.00%
     - 100.00%
     - 0.00%
     - 66.64%
     - 66.64%
   * - Amazon_Fashion
     - 100.00%
     - 100.00%
     - 0.00%
     - 76.58%
     - 76.58%
   * - Appliances
     - 100.00%
     - 100.00%
     - 0.00%
     - 5.40%
     - 5.40%
   * - Arts_Crafts_and_Sewing
     - 99.95%
     - 0.00%
     - 100.00%
     - 1.14%
     - 1.38%
   * - Automotive
     - >99.99%
     - >99.99%
     - 0.00%
     - 1.12%
     - 1.12%
   * - Baby_Products
     - 99.99%
     - 99.99%
     - 0.00%
     - 2.47%
     - 2.47%
   * - Beauty_and_Personal_Care
     - 100.00%
     - 100.00%
     - 0.00%
     - 3.24%
     - 3.24%
   * - Books
     - 88.66%
     - 88.66%
     - 0.00%
     - 1.37%
     - 1.37%
   * - CDs_and_Vinyl
     - 99.92%
     - 99.92%
     - 0.00%
     - 18.64%
     - 18.64%
   * - Cell_Phones_and_Accessories
     - 100.00%
     - 0.00%
     - 100.00%
     - 1.51%
     - 2.27%
   * - Clothing_Shoes_and_Jewelry
     - 99.99%
     - 99.99%
     - 0.00%
     - 0.40%
     - 0.40%
   * - Digital_Music
     - 100.00%
     - 100.00%
     - 0.00%
     - 72.51%
     - 72.51%
   * - Electronics
     - >99.99%
     - 0.00%
     - 100.00%
     - 2.06%
     - 2.34%
   * - Gift_Cards
     - 100.00%
     - 0.00%
     - 100.00%
     - 4.67%
     - 4.67%
   * - Grocery_and_Gourmet_Food
     - 99.95%
     - 99.95%
     - 0.00%
     - 2.92%
     - 2.92%
   * - Handmade_Products
     - 100.00%
     - 0.00%
     - 100.00%
     - 2.89%
     - 3.09%
   * - Health_and_Household
     - 100.00%
     - 100.00%
     - 0.00%
     - 0.73%
     - 0.73%
   * - Health_and_Personal_Care
     - 100.00%
     - 100.00%
     - 0.00%
     - 60.02%
     - 60.02%
   * - Home_and_Kitchen
     - 100.00%
     - 100.00%
     - 0.00%
     - 0.39%
     - 0.39%
   * - Industrial_and_Scientific
     - 100.00%
     - 0.00%
     - 100.00%
     - 1.48%
     - 1.69%
   * - Kindle_Store
     - 89.32%
     - 89.32%
     - 0.00%
     - 0.06%
     - 0.06%
   * - Magazine_Subscriptions
     - 100.00%
     - 100.00%
     - 0.00%
     - 60.11%
     - 60.11%
   * - Movies_and_TV
     - 100.00%
     - 47.63%
     - 0.00%
     - 50.09%
     - 50.09%
   * - Musical_Instruments
     - 99.98%
     - 0.00%
     - 100.00%
     - 1.90%
     - 2.10%
   * - Office_Products
     - 99.70%
     - 99.70%
     - 0.00%
     - 0.81%
     - 0.81%
   * - Patio_Lawn_and_Garden
     - 99.97%
     - 99.97%
     - 0.00%
     - 0.55%
     - 0.55%
   * - Pet_Supplies
     - 99.98%
     - 99.98%
     - 0.00%
     - 1.00%
     - 1.00%
   * - Software
     - >99.99%
     - >99.99%
     - 0.00%
     - 1.20%
     - 1.20%
   * - Sports_and_Outdoors
     - 100.00%
     - 100.00%
     - 0.00%
     - 1.01%
     - 1.01%
   * - Subscription_Boxes
     - 100.00%
     - 100.00%
     - 0.00%
     - 7.03%
     - 7.03%
   * - Tools_and_Home_Improvement
     - >99.99%
     - >99.99%
     - 0.00%
     - 0.57%
     - 0.57%
   * - Toys_and_Games
     - >99.99%
     - 0.00%
     - 100.00%
     - 0.35%
     - 0.38%
   * - Video_Games
     - >99.99%
     - >99.99%
     - 0.00%
     - 4.06%
     - 4.06%

.. _amazon-installed-text-recipes:

How to assemble text
--------------------

Keep the title as the compact identity anchor. Add features and descriptions when
present; they often complement each other rather than supplying interchangeable
coverage. Categories add taxonomy when present, but cannot compensate for missing
product-specific content. A store name can help identify brand, artist or author,
but must be inspected: it is not universally equivalent to any of those fields.

The subset recipes below are installed defaults, selected for semantic content
and coverage, not validated for recommendation quality. Missing optional fields
are skipped.
The order prioritizes store/creator context for media, then whichever prose field
has wider coverage; absent fields are omitted. All other nonempty optional prose
is kept. ``store`` needs category-specific cleanup: inspected Books entries include
author names and format labels, while other domains use brands or storefronts.
Inspected Books ``features`` include substantial synopses, so they must not be
discarded merely because the field name suggests short specification bullets.
There is no reason to change a split's support parameters merely to change field
ordering while ``min_entity_text_words=0``.

Movies and TV particularly needs curated nested attributes: many Prime Video
records have no title, features, description or category list, but do have
``details.Directors``, ``details.Starring``, audio languages and subtitles.
These provide a meaningful attribute fallback, not a recovered title or synopsis.
Just reordering the four existing fields cannot recover absent content.
Across the retained Movies and TV union, the original four-field text was nonempty
for only **53.65%** of items; the installed curated recipe reaches **100%**. Corrected
source image-URL coverage is also **100%**, while the current adapter exposes only
47.63%. Books and Kindle Store have genuine source-URL gaps: their coverage is
88.66% and 89.32%, respectively. None of these URLs has been fetched in this audit.

.. list-table:: Installed per-subset metadata_text_fields (both columns, in order)
   :header-rows: 1

   * - Subset
     - Top-level fields
     - Included nested detail fields
   * - All_Beauty
     - ``title,features,description,store``
     - ``details.Brand``; ``details.Item Form``; ``details.Skin Type``; ``details.Hair Type``; ``details.Product Benefits``
   * - Amazon_Fashion
     - ``title,features,description,store``
     - ``details.Department``; ``details.Brand``; ``details.Material``; ``details.Color``; ``details.Style``
   * - Appliances
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Compatible Devices``; ``details.Capacity``; ``details.Voltage``; ``details.Material``
   * - Arts_Crafts_and_Sewing
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Material``; ``details.Color``; ``details.Size``; ``details.Paint Type``
   * - Automotive
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Vehicle Service Type``; ``details.Auto Part Position``; ``details.Material``
   * - Baby_Products
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Age Range (Description)``; ``details.Material``; ``details.Style``
   * - Beauty_and_Personal_Care
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Item Form``; ``details.Skin Type``; ``details.Scent``; ``details.Product Benefits``
   * - Books
     - ``title,store,features,description,categories``
     - ``details.Publisher``; ``details.Language``; ``details.Paperback``; ``details.Hardcover``
   * - CDs_and_Vinyl
     - ``title,store,description,features,categories``
     - ``details.Label``; ``details.Language``; ``details.Media Format``; ``details.Run time``
   * - Cell_Phones_and_Accessories
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Compatible Devices``; ``details.Compatible Phone Models``; ``details.Material``
   * - Clothing_Shoes_and_Jewelry
     - ``title,features,description,categories,store``
     - ``details.Department``; ``details.Brand``; ``details.Material``; ``details.Color``; ``details.Style``
   * - Digital_Music
     - ``title,store,description,features,categories``
     - ``details.Label``; ``details.Genres``; ``details.Language``; ``details.Run time``
   * - Electronics
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Compatible Devices``; ``details.Connectivity Technology``; ``details.Screen Size``; ``details.Memory Storage Capacity``
   * - Gift_Cards
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Manufacturer``
   * - Grocery_and_Gourmet_Food
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Flavor``; ``details.Item Form``; ``details.Specialty``; ``details.Allergen Information``
   * - Handmade_Products
     - ``title,description,features,categories,store``
     - ``details.Material``; ``details.Color``; ``details.Size``; ``details.Department``
   * - Health_and_Household
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Item Form``; ``details.Active Ingredients``; ``details.Material``; ``details.Scent``
   * - Health_and_Personal_Care
     - ``title,features,description,store``
     - ``details.Brand``; ``details.Item Form``; ``details.Active Ingredients``; ``details.Skin Type``
   * - Home_and_Kitchen
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Material``; ``details.Color``; ``details.Style``; ``details.Size``
   * - Industrial_and_Scientific
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Material``; ``details.Size``; ``details.Compatible Devices``
   * - Kindle_Store
     - ``title,store,features,description,categories``
     - ``details.Publisher``; ``details.Language``; ``details.Print length``; ``details.Author``
   * - Magazine_Subscriptions
     - ``title,store,description,categories``
     - ``details.Publisher``; ``details.Language``; ``details.Print length``
   * - Movies_and_TV
     - ``title,store,description,features,categories``
     - ``details.Director``; ``details.Directors``; ``details.Actors``; ``details.Starring``; ``details.Media Format``; ``details.Language``; ``details.Audio languages``; ``details.Subtitles``; ``details.Run time``
   * - Musical_Instruments
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Instrument``; ``details.Material``; ``details.Compatible Devices``
   * - Office_Products
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Material``; ``details.Color``; ``details.Sheet Size``; ``details.Ink Color``
   * - Patio_Lawn_and_Garden
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Material``; ``details.Power Source``; ``details.Color``
   * - Pet_Supplies
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Target Species``; ``details.Breed Recommendation``; ``details.Flavor``; ``details.Material``
   * - Software
     - ``title,store,features,description,categories``
     - ``details.Platform``; ``details.Operating System``; ``details.Manufacturer``; ``details.Language``
   * - Sports_and_Outdoors
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Sport``; ``details.Material``; ``details.Size``; ``details.Age Range (Description)``
   * - Subscription_Boxes
     - ``title,features,store``
     - ``details.Brand``; ``details.Manufacturer``; ``details.Material``
   * - Tools_and_Home_Improvement
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Material``; ``details.Power Source``; ``details.Voltage``
   * - Toys_and_Games
     - ``title,features,description,categories,store``
     - ``details.Brand``; ``details.Age Range (Description)``; ``details.Material``; ``details.Theme``; ``details.Educational Objective``
   * - Video_Games
     - ``title,store,features,description,categories``
     - ``details.Platform``; ``details.Rated``; ``details.Manufacturer``; ``details.Language``

Do not concatenate all ``details`` indiscriminately. The measured dictionaries
include useful category-specific attributes, but also identifiers, packaging,
dates and, in many subsets, **Best Sellers Rank**. Exclude popularity aggregates
(``average_rating``, ``rating_number`` and ranks), IDs such as ASIN/ISBN/UPC,
purchase/co-purchase links, and irrelevant shipping boilerplate from semantic
text. Snapshot popularity can reveal post-split information and confound a
content-only or temporal experiment. Even ordinary descriptions are crawl-time
snapshots, not historically versioned metadata.

The recipe table lists the **full installed selection**, not only frequently
observed keys. Both columns form one ordered ``metadata_text_fields`` tuple.
The same category recipe applies to ``user_split``, ``item_split``,
``leave_last_out`` and ``temporal``; it does not modify their support thresholds,
holdout sizes or rating policy. Unknown/unprofiled categories fall back to
``title,features,description,categories``. Existing checkpoints are not rewritten;
rebuild one to obtain the enriched ``entity_text``. The resolved field list is
recorded in the new checkpoint's data-stage manifest.

To replace the default explicitly, for example:

.. code-block:: python

   cr.build_recsys_checkpoint(
       dataset="amazon2023",
       amazon_category="Movies_and_TV",
       metadata_text_fields=["title", "description", "details.Directors", "details.Starring"],
       min_entity_text_words=0,
   )

An explicit top-level ``details`` still includes the entire dictionary. That is
not the default: it can raise word counts with identifiers or popularity data.
Further whitespace/HTML cleanup, deduplication and encoder truncation remain
embedding-pipeline recommendations, not part of these measured word counts.

The next table compares the original four-field baseline with the **installed
curated recipe**: normalized text containers, available prose/taxonomy fields,
``store`` where observed, and the category's allowlisted details. It uses the
original audit's curated counts; regression tests check equivalent word counts
for all 33 recipes. Field order alone does not change these counts. More nonempty
or longer text does not itself prove better recommendation quality.

.. list-table:: Measured effect of the installed curated defaults
   :header-rows: 1

   * - Subset
     - Original four-field text coverage
     - Installed text coverage
     - Installed items below 30
     - Original train-pair loss at 30: range over splits
     - Installed train-pair loss at 30: range over splits
   * - All_Beauty
     - >99.99%
     - 100.00%
     - 40.47%
     - 56.21%–63.96%
     - 33.50%–35.27%
   * - Amazon_Fashion
     - >99.99%
     - 100.00%
     - 72.44%
     - 66.35%–75.36%
     - 60.25%–70.13%
   * - Appliances
     - 100.00%
     - 100.00%
     - 3.56%
     - 1.93%–2.29%
     - 1.27%–1.50%
   * - Arts_Crafts_and_Sewing
     - 100.00%
     - 100.00%
     - 0.69%
     - 0.71%–0.98%
     - 0.42%–0.57%
   * - Automotive
     - 100.00%
     - 100.00%
     - 0.45%
     - 0.68%–0.79%
     - 0.27%–0.32%
   * - Baby_Products
     - 100.00%
     - 100.00%
     - 1.36%
     - 0.87%–1.31%
     - 0.40%–0.67%
   * - Beauty_and_Personal_Care
     - 100.00%
     - 100.00%
     - 0.72%
     - 2.55%–3.01%
     - 0.61%–0.72%
   * - Books
     - 100.00%
     - 100.00%
     - 0.39%
     - 0.54%–0.64%
     - 0.14%–0.18%
   * - CDs_and_Vinyl
     - 100.00%
     - 100.00%
     - 8.43%
     - 13.63%–15.02%
     - 6.42%–6.61%
   * - Cell_Phones_and_Accessories
     - 100.00%
     - 100.00%
     - 0.74%
     - 0.78%–1.21%
     - 0.34%–0.58%
   * - Clothing_Shoes_and_Jewelry
     - 100.00%
     - 100.00%
     - 0.14%
     - 0.40%–0.45%
     - 0.07%–0.10%
   * - Digital_Music
     - >99.99%
     - 100.00%
     - 63.81%
     - 45.40%–72.12%
     - 36.12%–62.42%
   * - Electronics
     - 100.00%
     - 100.00%
     - 1.16%
     - 0.97%–1.27%
     - 0.61%–0.76%
   * - Gift_Cards
     - 100.00%
     - 100.00%
     - 3.74%
     - 0.97%–1.19%
     - 0.74%–0.88%
   * - Grocery_and_Gourmet_Food
     - 100.00%
     - 100.00%
     - 1.30%
     - 2.06%–2.16%
     - 0.87%–0.95%
   * - Handmade_Products
     - 100.00%
     - 100.00%
     - 2.13%
     - 1.39%–2.56%
     - 0.94%–1.81%
   * - Health_and_Household
     - 100.00%
     - 100.00%
     - 0.16%
     - 0.54%–0.65%
     - 0.13%–0.15%
   * - Health_and_Personal_Care
     - 99.98%
     - 100.00%
     - 45.14%
     - 48.57%–53.01%
     - 30.43%–34.11%
   * - Home_and_Kitchen
     - 100.00%
     - 100.00%
     - 0.18%
     - 0.16%–0.24%
     - 0.07%–0.11%
   * - Industrial_and_Scientific
     - 100.00%
     - 100.00%
     - 0.86%
     - 0.73%–0.96%
     - 0.36%–0.50%
   * - Kindle_Store
     - 100.00%
     - 100.00%
     - <0.01%
     - 0.07%–0.09%
     - 0.02%–0.02%
   * - Magazine_Subscriptions
     - 100.00%
     - 100.00%
     - 57.96%
     - 59.52%–63.15%
     - 59.04%–62.26%
   * - Movies_and_TV
     - 53.65%
     - 100.00%
     - 33.02%
     - 51.54%–62.21%
     - 26.62%–36.33%
   * - Musical_Instruments
     - 100.00%
     - 100.00%
     - 1.49%
     - 0.88%–1.10%
     - 0.62%–0.78%
   * - Office_Products
     - 100.00%
     - 100.00%
     - 0.33%
     - 0.37%–0.47%
     - 0.21%–0.26%
   * - Patio_Lawn_and_Garden
     - 100.00%
     - 100.00%
     - 0.14%
     - 0.43%–0.54%
     - 0.12%–0.14%
   * - Pet_Supplies
     - >99.99%
     - 100.00%
     - 0.49%
     - 0.41%–0.56%
     - 0.15%–0.22%
   * - Software
     - 100.00%
     - 100.00%
     - 0.88%
     - 0.47%–0.80%
     - 0.39%–0.71%
   * - Sports_and_Outdoors
     - 100.00%
     - 100.00%
     - 0.40%
     - 0.52%–0.71%
     - 0.20%–0.30%
   * - Subscription_Boxes
     - 99.69%
     - 99.69%
     - 7.03%
     - 4.51%–8.82%
     - 4.51%–8.82%
   * - Tools_and_Home_Improvement
     - 100.00%
     - 100.00%
     - 0.21%
     - 0.32%–0.41%
     - 0.11%–0.15%
   * - Toys_and_Games
     - 100.00%
     - 100.00%
     - 0.18%
     - 0.17%–0.20%
     - 0.05%–0.10%
   * - Video_Games
     - 100.00%
     - 100.00%
     - 2.80%
     - 2.18%–2.28%
     - 1.63%–1.72%

For limited encoder context, preserve title and selected category-specific
attributes, then allocate a budget to features and description. Use the actual
encoder tokenizer to set that budget; whitespace word counts are not token counts.
Record field order, normalization, selected attribute keys, encoder and truncation
policy alongside embeddings so experiments remain reproducible.

Recommendation
--------------

Keep **min_entity_text_words=0** for the common collaborative/multimodal benchmark.
Preserve the verified interactions and splits, then record text/image availability
masks with embeddings. Evaluate content-only methods on explicitly defined
available-feature cohorts and report excluded coverage, or use a documented
fallback. Do not silently drop users or items only for one competing method.

If a text-only benchmark genuinely requires a minimum length, choose it per
subset from the measured loss curves, rebuild support pruning and every split,
and recheck eligible validation/test users. A threshold of 1 tests for some
combined text; neither 1 nor 30 guarantees a useful description. Real descriptions
should be checked directly when that is the scientific requirement.

Detailed split measurements
---------------------------

Text coverage and direct threshold losses below use the installed curated
recipes. With the default threshold of zero, no interactions are removed for
text length; the user/item counts and verified split profiles are unchanged.

.. list-table:: Per-split installed-default coverage and direct training-pair loss
   :header-rows: 1

   * - Subset
     - Split
     - Catalog: items / text / image
     - Train: items / text / image
     - Validation targets: items / text / image
     - Test targets: items / text / image
     - Train pairs lost below 1 / 10 / 30 words
   * - All_Beauty
     - user_split
     - 11,834 / 100.00% / 100.00%
     - 11,834 / 100.00% / 100.00%
     - 1,826 / 100.00% / 100.00%
     - 1,854 / 100.00% / 100.00%
     - 0.00% / 1.01% / 34.73%
   * - All_Beauty
     - item_split
     - 12,105 / 100.00% / 100.00%
     - 10,288 / 100.00% / 100.00%
     - 602 / 100.00% / 100.00%
     - 1,203 / 100.00% / 100.00%
     - 0.00% / 1.01% / 35.27%
   * - All_Beauty
     - leave_last_out
     - 10,266 / 100.00% / 100.00%
     - 7,242 / 100.00% / 100.00%
     - 2,503 / 100.00% / 100.00%
     - 2,487 / 100.00% / 100.00%
     - 0.00% / 0.99% / 34.46%
   * - All_Beauty
     - temporal
     - 10,734 / 100.00% / 100.00%
     - 7,173 / 100.00% / 100.00%
     - 1,657 / 100.00% / 100.00%
     - 618 / 100.00% / 100.00%
     - 0.00% / 0.94% / 33.50%
   * - Amazon_Fashion
     - user_split
     - 10,206 / 100.00% / 100.00%
     - 10,206 / 100.00% / 100.00%
     - 2,779 / 100.00% / 100.00%
     - 2,692 / 100.00% / 100.00%
     - 0.00% / 0.76% / 64.26%
   * - Amazon_Fashion
     - item_split
     - 10,217 / 100.00% / 100.00%
     - 8,684 / 100.00% / 100.00%
     - 511 / 100.00% / 100.00%
     - 1,022 / 100.00% / 100.00%
     - 0.00% / 0.73% / 64.24%
   * - Amazon_Fashion
     - leave_last_out
     - 4,098 / 100.00% / 100.00%
     - 3,788 / 100.00% / 100.00%
     - 1,292 / 100.00% / 100.00%
     - 1,190 / 100.00% / 100.00%
     - 0.00% / 0.91% / 70.13%
   * - Amazon_Fashion
     - temporal
     - 4,080 / 100.00% / 100.00%
     - 2,933 / 100.00% / 100.00%
     - 673 / 100.00% / 100.00%
     - 467 / 100.00% / 100.00%
     - 0.00% / 0.90% / 60.25%
   * - Appliances
     - user_split
     - 19,078 / 100.00% / 100.00%
     - 19,078 / 100.00% / 100.00%
     - 3,391 / 100.00% / 100.00%
     - 3,277 / 100.00% / 100.00%
     - 0.00% / <0.01% / 1.46%
   * - Appliances
     - item_split
     - 19,327 / 100.00% / 100.00%
     - 16,427 / 100.00% / 100.00%
     - 967 / 100.00% / 100.00%
     - 1,933 / 100.00% / 100.00%
     - 0.00% / <0.01% / 1.49%
   * - Appliances
     - leave_last_out
     - 10,896 / 100.00% / 100.00%
     - 9,603 / 100.00% / 100.00%
     - 6,274 / 100.00% / 100.00%
     - 6,215 / 100.00% / 100.00%
     - 0.00% / <0.01% / 1.50%
   * - Appliances
     - temporal
     - 16,820 / 100.00% / 100.00%
     - 10,232 / 100.00% / 100.00%
     - 8,307 / 100.00% / 100.00%
     - 7,199 / 100.00% / 100.00%
     - 0.00% / 0.01% / 1.27%
   * - Arts_Crafts_and_Sewing
     - user_split
     - 18,043 / 100.00% / 99.95%
     - 18,043 / 100.00% / 99.95%
     - 6,506 / 100.00% / 99.97%
     - 6,613 / 100.00% / 99.95%
     - 0.00% / <0.01% / 0.53%
   * - Arts_Crafts_and_Sewing
     - item_split
     - 18,043 / 100.00% / 99.95%
     - 15,335 / 100.00% / 99.94%
     - 903 / 100.00% / 100.00%
     - 1,805 / 100.00% / 100.00%
     - 0.00% / 0.01% / 0.49%
   * - Arts_Crafts_and_Sewing
     - leave_last_out
     - 18,043 / 100.00% / 99.95%
     - 18,043 / 100.00% / 99.95%
     - 16,863 / 100.00% / 99.95%
     - 16,459 / 100.00% / 99.97%
     - 0.00% / <0.01% / 0.57%
   * - Arts_Crafts_and_Sewing
     - temporal
     - 12,713 / 100.00% / 99.97%
     - 8,849 / 100.00% / 99.97%
     - 9,307 / 100.00% / 99.97%
     - 9,221 / 100.00% / 99.98%
     - 0.00% / <0.01% / 0.42%
   * - Automotive
     - user_split
     - 16,079 / 100.00% / >99.99%
     - 16,079 / 100.00% / >99.99%
     - 7,492 / 100.00% / 99.99%
     - 7,574 / 100.00% / 99.99%
     - 0.00% / <0.01% / 0.30%
   * - Automotive
     - item_split
     - 16,079 / 100.00% / >99.99%
     - 13,667 / 100.00% / >99.99%
     - 804 / 100.00% / 100.00%
     - 1,608 / 100.00% / 100.00%
     - 0.00% / <0.01% / 0.28%
   * - Automotive
     - leave_last_out
     - 16,079 / 100.00% / >99.99%
     - 16,079 / 100.00% / >99.99%
     - 15,101 / 100.00% / >99.99%
     - 14,639 / 100.00% / >99.99%
     - 0.00% / <0.01% / 0.32%
   * - Automotive
     - temporal
     - 19,182 / 100.00% / >99.99%
     - 13,869 / 100.00% / >99.99%
     - 15,181 / 100.00% / >99.99%
     - 12,891 / 100.00% / >99.99%
     - 0.00% / <0.01% / 0.27%
   * - Baby_Products
     - user_split
     - 17,987 / 100.00% / 99.98%
     - 17,987 / 100.00% / 99.98%
     - 6,029 / 100.00% / 99.98%
     - 6,056 / 100.00% / 99.98%
     - 0.00% / 0.01% / 0.65%
   * - Baby_Products
     - item_split
     - 17,987 / 100.00% / 99.98%
     - 15,288 / 100.00% / 99.99%
     - 900 / 100.00% / 100.00%
     - 1,799 / 100.00% / 99.94%
     - 0.00% / <0.01% / 0.57%
   * - Baby_Products
     - leave_last_out
     - 17,987 / 100.00% / 99.98%
     - 17,983 / 100.00% / 99.98%
     - 14,115 / 100.00% / 99.99%
     - 13,560 / 100.00% / 99.98%
     - 0.00% / 0.01% / 0.67%
   * - Baby_Products
     - temporal
     - 13,365 / 100.00% / 99.98%
     - 9,096 / 100.00% / 99.97%
     - 7,822 / 100.00% / 99.97%
     - 8,198 / 100.00% / 99.98%
     - 0.00% / <0.01% / 0.40%
   * - Beauty_and_Personal_Care
     - user_split
     - 16,817 / 100.00% / 100.00%
     - 16,817 / 100.00% / 100.00%
     - 7,981 / 100.00% / 100.00%
     - 8,008 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.69%
   * - Beauty_and_Personal_Care
     - item_split
     - 16,817 / 100.00% / 100.00%
     - 14,294 / 100.00% / 100.00%
     - 841 / 100.00% / 100.00%
     - 1,682 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.63%
   * - Beauty_and_Personal_Care
     - leave_last_out
     - 16,817 / 100.00% / 100.00%
     - 16,817 / 100.00% / 100.00%
     - 15,180 / 100.00% / 100.00%
     - 14,750 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.72%
   * - Beauty_and_Personal_Care
     - temporal
     - 14,656 / 100.00% / 100.00%
     - 8,247 / 100.00% / 100.00%
     - 9,685 / 100.00% / 100.00%
     - 11,789 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.61%
   * - Books
     - user_split
     - 94,864 / 100.00% / 89.17%
     - 94,864 / 100.00% / 89.17%
     - 31,567 / 100.00% / 89.06%
     - 32,453 / 100.00% / 89.01%
     - 0.00% / 0.01% / 0.16%
   * - Books
     - item_split
     - 94,864 / 100.00% / 89.17%
     - 80,633 / 100.00% / 89.14%
     - 4,744 / 100.00% / 88.47%
     - 9,487 / 100.00% / 89.73%
     - 0.00% / 0.01% / 0.16%
   * - Books
     - leave_last_out
     - 94,864 / 100.00% / 89.17%
     - 94,829 / 100.00% / 89.17%
     - 80,327 / 100.00% / 89.17%
     - 76,778 / 100.00% / 89.13%
     - 0.00% / 0.01% / 0.14%
   * - Books
     - temporal
     - 96,465 / 100.00% / 88.36%
     - 75,631 / 100.00% / 88.73%
     - 37,923 / 100.00% / 88.00%
     - 26,194 / 100.00% / 87.40%
     - 0.00% / 0.02% / 0.18%
   * - CDs_and_Vinyl
     - user_split
     - 18,586 / 100.00% / 99.90%
     - 18,586 / 100.00% / 99.90%
     - 8,295 / 100.00% / 99.89%
     - 8,493 / 100.00% / 99.89%
     - 0.00% / 0.00% / 6.42%
   * - CDs_and_Vinyl
     - item_split
     - 18,586 / 100.00% / 99.90%
     - 15,797 / 100.00% / 99.91%
     - 930 / 100.00% / 99.68%
     - 1,859 / 100.00% / 100.00%
     - 0.00% / 0.00% / 6.44%
   * - CDs_and_Vinyl
     - leave_last_out
     - 18,586 / 100.00% / 99.90%
     - 18,584 / 100.00% / 99.90%
     - 16,271 / 100.00% / 99.90%
     - 16,039 / 100.00% / 99.89%
     - 0.00% / 0.00% / 6.44%
   * - CDs_and_Vinyl
     - temporal
     - 13,650 / 100.00% / 99.88%
     - 11,048 / 100.00% / 99.86%
     - 6,986 / 100.00% / 99.86%
     - 4,566 / 100.00% / 99.82%
     - 0.00% / 0.00% / 6.61%
   * - Cell_Phones_and_Accessories
     - user_split
     - 19,322 / 100.00% / 100.00%
     - 19,322 / 100.00% / 100.00%
     - 6,692 / 100.00% / 100.00%
     - 6,736 / 100.00% / 100.00%
     - 0.00% / 0.01% / 0.50%
   * - Cell_Phones_and_Accessories
     - item_split
     - 19,322 / 100.00% / 100.00%
     - 16,422 / 100.00% / 100.00%
     - 967 / 100.00% / 100.00%
     - 1,933 / 100.00% / 100.00%
     - 0.00% / 0.01% / 0.51%
   * - Cell_Phones_and_Accessories
     - leave_last_out
     - 19,322 / 100.00% / 100.00%
     - 19,315 / 100.00% / 100.00%
     - 16,504 / 100.00% / 100.00%
     - 15,069 / 100.00% / 100.00%
     - 0.00% / 0.01% / 0.58%
   * - Cell_Phones_and_Accessories
     - temporal
     - 19,096 / 100.00% / 100.00%
     - 10,107 / 100.00% / 100.00%
     - 7,932 / 100.00% / 100.00%
     - 10,338 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.34%
   * - Clothing_Shoes_and_Jewelry
     - user_split
     - 14,289 / 100.00% / 99.98%
     - 14,289 / 100.00% / 99.98%
     - 8,145 / 100.00% / 100.00%
     - 8,166 / 100.00% / 99.98%
     - 0.00% / 0.00% / 0.08%
   * - Clothing_Shoes_and_Jewelry
     - item_split
     - 14,289 / 100.00% / 99.98%
     - 12,145 / 100.00% / 99.98%
     - 715 / 100.00% / 100.00%
     - 1,429 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.07%
   * - Clothing_Shoes_and_Jewelry
     - leave_last_out
     - 14,289 / 100.00% / 99.98%
     - 14,289 / 100.00% / 99.98%
     - 13,220 / 100.00% / 99.98%
     - 12,877 / 100.00% / 99.98%
     - 0.00% / 0.00% / 0.09%
   * - Clothing_Shoes_and_Jewelry
     - temporal
     - 24,824 / 100.00% / 99.99%
     - 10,351 / 100.00% / 99.97%
     - 20,663 / 100.00% / >99.99%
     - 19,421 / 100.00% / >99.99%
     - 0.00% / 0.00% / 0.10%
   * - Digital_Music
     - user_split
     - 2,341 / 100.00% / 100.00%
     - 2,341 / 100.00% / 100.00%
     - 252 / 100.00% / 100.00%
     - 240 / 100.00% / 100.00%
     - 0.00% / 3.52% / 36.12%
   * - Digital_Music
     - item_split
     - 2,389 / 100.00% / 100.00%
     - 2,030 / 100.00% / 100.00%
     - 119 / 100.00% / 100.00%
     - 236 / 100.00% / 100.00%
     - 0.00% / 3.58% / 36.70%
   * - Digital_Music
     - leave_last_out
     - 16,240 / 100.00% / 100.00%
     - 12,553 / 100.00% / 100.00%
     - 2,184 / 100.00% / 100.00%
     - 2,177 / 100.00% / 100.00%
     - 0.00% / 10.79% / 62.42%
   * - Digital_Music
     - temporal
     - 8,154 / 100.00% / 100.00%
     - 4,145 / 100.00% / 100.00%
     - 1,121 / 100.00% / 100.00%
     - 735 / 100.00% / 100.00%
     - 0.00% / 8.47% / 61.69%
   * - Electronics
     - user_split
     - 92,180 / 100.00% / >99.99%
     - 92,180 / 100.00% / >99.99%
     - 31,288 / 100.00% / >99.99%
     - 31,489 / 100.00% / >99.99%
     - 0.00% / 0.06% / 0.71%
   * - Electronics
     - item_split
     - 92,180 / 100.00% / >99.99%
     - 78,353 / 100.00% / >99.99%
     - 4,609 / 100.00% / 100.00%
     - 9,218 / 100.00% / 100.00%
     - 0.00% / 0.06% / 0.74%
   * - Electronics
     - leave_last_out
     - 92,180 / 100.00% / >99.99%
     - 92,177 / 100.00% / >99.99%
     - 72,119 / 100.00% / >99.99%
     - 66,509 / 100.00% / >99.99%
     - 0.00% / 0.06% / 0.76%
   * - Electronics
     - temporal
     - 78,239 / 100.00% / >99.99%
     - 58,356 / 100.00% / >99.99%
     - 39,800 / 100.00% / >99.99%
     - 40,647 / 100.00% / 100.00%
     - 0.00% / 0.06% / 0.61%
   * - Gift_Cards
     - user_split
     - 717 / 100.00% / 100.00%
     - 717 / 100.00% / 100.00%
     - 230 / 100.00% / 100.00%
     - 240 / 100.00% / 100.00%
     - 0.00% / <0.01% / 0.74%
   * - Gift_Cards
     - item_split
     - 575 / 100.00% / 100.00%
     - 488 / 100.00% / 100.00%
     - 29 / 100.00% / 100.00%
     - 58 / 100.00% / 100.00%
     - 0.00% / <0.01% / 0.80%
   * - Gift_Cards
     - leave_last_out
     - 503 / 100.00% / 100.00%
     - 439 / 100.00% / 100.00%
     - 253 / 100.00% / 100.00%
     - 244 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.88%
   * - Gift_Cards
     - temporal
     - 421 / 100.00% / 100.00%
     - 278 / 100.00% / 100.00%
     - 123 / 100.00% / 100.00%
     - 120 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.88%
   * - Grocery_and_Gourmet_Food
     - user_split
     - 16,735 / 100.00% / 99.94%
     - 16,735 / 100.00% / 99.94%
     - 7,768 / 100.00% / 99.96%
     - 7,813 / 100.00% / 99.95%
     - 0.00% / 0.01% / 0.90%
   * - Grocery_and_Gourmet_Food
     - item_split
     - 16,735 / 100.00% / 99.94%
     - 14,224 / 100.00% / 99.94%
     - 837 / 100.00% / 100.00%
     - 1,674 / 100.00% / 99.94%
     - 0.00% / <0.01% / 0.87%
   * - Grocery_and_Gourmet_Food
     - leave_last_out
     - 16,735 / 100.00% / 99.94%
     - 16,735 / 100.00% / 99.94%
     - 15,347 / 100.00% / 99.93%
     - 14,851 / 100.00% / 99.94%
     - 0.00% / 0.01% / 0.92%
   * - Grocery_and_Gourmet_Food
     - temporal
     - 19,238 / 100.00% / 99.96%
     - 12,329 / 100.00% / 99.97%
     - 14,478 / 100.00% / 99.97%
     - 13,429 / 100.00% / 99.94%
     - 0.00% / 0.02% / 0.95%
   * - Handmade_Products
     - user_split
     - 11,613 / 100.00% / 100.00%
     - 11,613 / 100.00% / 100.00%
     - 1,725 / 100.00% / 100.00%
     - 1,788 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.95%
   * - Handmade_Products
     - item_split
     - 11,878 / 100.00% / 100.00%
     - 10,096 / 100.00% / 100.00%
     - 587 / 100.00% / 100.00%
     - 1,180 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.94%
   * - Handmade_Products
     - leave_last_out
     - 14,723 / 100.00% / 100.00%
     - 9,726 / 100.00% / 100.00%
     - 3,350 / 100.00% / 100.00%
     - 3,347 / 100.00% / 100.00%
     - 0.00% / 0.00% / 1.81%
   * - Handmade_Products
     - temporal
     - 11,710 / 100.00% / 100.00%
     - 5,138 / 100.00% / 100.00%
     - 2,029 / 100.00% / 100.00%
     - 1,708 / 100.00% / 100.00%
     - 0.00% / 0.00% / 1.77%
   * - Health_and_Household
     - user_split
     - 19,922 / 100.00% / 100.00%
     - 19,922 / 100.00% / 100.00%
     - 9,180 / 100.00% / 100.00%
     - 8,986 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.13%
   * - Health_and_Household
     - item_split
     - 19,922 / 100.00% / 100.00%
     - 16,932 / 100.00% / 100.00%
     - 997 / 100.00% / 100.00%
     - 1,993 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.15%
   * - Health_and_Household
     - leave_last_out
     - 19,922 / 100.00% / 100.00%
     - 19,922 / 100.00% / 100.00%
     - 16,439 / 100.00% / 100.00%
     - 15,703 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.13%
   * - Health_and_Household
     - temporal
     - 19,907 / 100.00% / 100.00%
     - 12,277 / 100.00% / 100.00%
     - 13,166 / 100.00% / 100.00%
     - 14,545 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.14%
   * - Health_and_Personal_Care
     - user_split
     - 15,041 / 100.00% / 100.00%
     - 15,041 / 100.00% / 100.00%
     - 940 / 100.00% / 100.00%
     - 877 / 100.00% / 100.00%
     - 0.00% / 1.65% / 34.11%
   * - Health_and_Personal_Care
     - item_split
     - 17,288 / 100.00% / 100.00%
     - 14,694 / 100.00% / 100.00%
     - 784 / 100.00% / 100.00%
     - 1,585 / 100.00% / 100.00%
     - 0.00% / 1.58% / 34.09%
   * - Health_and_Personal_Care
     - leave_last_out
     - 3,448 / 100.00% / 100.00%
     - 2,697 / 100.00% / 100.00%
     - 911 / 100.00% / 100.00%
     - 857 / 100.00% / 100.00%
     - 0.00% / 0.99% / 30.43%
   * - Health_and_Personal_Care
     - temporal
     - 7,604 / 100.00% / 100.00%
     - 4,625 / 100.00% / 100.00%
     - 1,603 / 100.00% / 100.00%
     - 762 / 100.00% / 100.00%
     - 0.00% / 1.16% / 32.96%
   * - Home_and_Kitchen
     - user_split
     - 17,431 / 100.00% / 100.00%
     - 17,431 / 100.00% / 100.00%
     - 9,721 / 100.00% / 100.00%
     - 9,750 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.10%
   * - Home_and_Kitchen
     - item_split
     - 17,431 / 100.00% / 100.00%
     - 14,815 / 100.00% / 100.00%
     - 872 / 100.00% / 100.00%
     - 1,744 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.10%
   * - Home_and_Kitchen
     - leave_last_out
     - 17,431 / 100.00% / 100.00%
     - 17,431 / 100.00% / 100.00%
     - 15,181 / 100.00% / 100.00%
     - 14,374 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.11%
   * - Home_and_Kitchen
     - temporal
     - 14,879 / 100.00% / 100.00%
     - 11,240 / 100.00% / 100.00%
     - 11,264 / 100.00% / 100.00%
     - 10,726 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.07%
   * - Industrial_and_Scientific
     - user_split
     - 19,251 / 100.00% / 100.00%
     - 19,251 / 100.00% / 100.00%
     - 5,348 / 100.00% / 100.00%
     - 5,350 / 100.00% / 100.00%
     - 0.00% / <0.01% / 0.45%
   * - Industrial_and_Scientific
     - item_split
     - 19,251 / 100.00% / 100.00%
     - 16,362 / 100.00% / 100.00%
     - 963 / 100.00% / 100.00%
     - 1,926 / 100.00% / 100.00%
     - 0.00% / 0.01% / 0.46%
   * - Industrial_and_Scientific
     - leave_last_out
     - 19,251 / 100.00% / 100.00%
     - 19,229 / 100.00% / 100.00%
     - 16,742 / 100.00% / 100.00%
     - 16,000 / 100.00% / 100.00%
     - 0.00% / 0.01% / 0.50%
   * - Industrial_and_Scientific
     - temporal
     - 17,690 / 100.00% / 100.00%
     - 10,270 / 100.00% / 100.00%
     - 10,279 / 100.00% / 100.00%
     - 10,034 / 100.00% / 100.00%
     - 0.00% / 0.01% / 0.36%
   * - Kindle_Store
     - user_split
     - 17,800 / 100.00% / 89.30%
     - 17,800 / 100.00% / 89.30%
     - 13,209 / 100.00% / 89.36%
     - 13,406 / 100.00% / 89.39%
     - 0.00% / 0.00% / 0.02%
   * - Kindle_Store
     - item_split
     - 17,800 / 100.00% / 89.30%
     - 15,130 / 100.00% / 89.25%
     - 890 / 100.00% / 90.67%
     - 1,780 / 100.00% / 89.10%
     - 0.00% / 0.00% / 0.02%
   * - Kindle_Store
     - leave_last_out
     - 17,800 / 100.00% / 89.30%
     - 17,800 / 100.00% / 89.30%
     - 15,961 / 100.00% / 89.34%
     - 15,154 / 100.00% / 89.34%
     - 0.00% / 0.00% / 0.02%
   * - Kindle_Store
     - temporal
     - 19,957 / 100.00% / 89.28%
     - 16,217 / 100.00% / 89.28%
     - 15,954 / 100.00% / 89.31%
     - 14,155 / 100.00% / 89.21%
     - 0.00% / 0.00% / 0.02%
   * - Magazine_Subscriptions
     - user_split
     - 1,708 / 100.00% / 100.00%
     - 1,708 / 100.00% / 100.00%
     - 472 / 100.00% / 100.00%
     - 463 / 100.00% / 100.00%
     - 0.00% / 4.54% / 61.56%
   * - Magazine_Subscriptions
     - item_split
     - 1,241 / 100.00% / 100.00%
     - 1,053 / 100.00% / 100.00%
     - 63 / 100.00% / 100.00%
     - 124 / 100.00% / 100.00%
     - 0.00% / 5.08% / 61.97%
   * - Magazine_Subscriptions
     - leave_last_out
     - 1,110 / 100.00% / 100.00%
     - 918 / 100.00% / 100.00%
     - 353 / 100.00% / 100.00%
     - 381 / 100.00% / 100.00%
     - 0.00% / 4.57% / 62.26%
   * - Magazine_Subscriptions
     - temporal
     - 522 / 100.00% / 100.00%
     - 329 / 100.00% / 100.00%
     - 117 / 100.00% / 100.00%
     - 66 / 100.00% / 100.00%
     - 0.00% / 4.34% / 59.04%
   * - Movies_and_TV
     - user_split
     - 18,324 / 100.00% / 100.00%
     - 18,324 / 100.00% / 100.00%
     - 10,254 / 100.00% / 100.00%
     - 10,070 / 100.00% / 100.00%
     - 0.00% / 0.44% / 26.86%
   * - Movies_and_TV
     - item_split
     - 18,324 / 100.00% / 100.00%
     - 15,574 / 100.00% / 100.00%
     - 917 / 100.00% / 100.00%
     - 1,833 / 100.00% / 100.00%
     - 0.00% / 0.34% / 26.72%
   * - Movies_and_TV
     - leave_last_out
     - 18,324 / 100.00% / 100.00%
     - 18,324 / 100.00% / 100.00%
     - 16,454 / 100.00% / 100.00%
     - 15,927 / 100.00% / 100.00%
     - 0.00% / 0.45% / 26.62%
   * - Movies_and_TV
     - temporal
     - 17,522 / 100.00% / 100.00%
     - 14,804 / 100.00% / 100.00%
     - 10,612 / 100.00% / 100.00%
     - 6,941 / 100.00% / 100.00%
     - 0.00% / 0.52% / 36.33%
   * - Musical_Instruments
     - user_split
     - 16,766 / 100.00% / 99.98%
     - 16,766 / 100.00% / 99.98%
     - 5,933 / 100.00% / 99.97%
     - 5,863 / 100.00% / 99.98%
     - 0.00% / 0.02% / 0.73%
   * - Musical_Instruments
     - item_split
     - 16,766 / 100.00% / 99.98%
     - 14,250 / 100.00% / 99.97%
     - 839 / 100.00% / 100.00%
     - 1,677 / 100.00% / 100.00%
     - 0.00% / 0.02% / 0.71%
   * - Musical_Instruments
     - leave_last_out
     - 16,766 / 100.00% / 99.98%
     - 16,760 / 100.00% / 99.98%
     - 12,985 / 100.00% / 99.97%
     - 12,681 / 100.00% / 99.98%
     - 0.00% / 0.02% / 0.78%
   * - Musical_Instruments
     - temporal
     - 15,914 / 100.00% / 99.98%
     - 11,545 / 100.00% / >99.99%
     - 10,123 / 100.00% / >99.99%
     - 9,093 / 100.00% / 99.98%
     - 0.00% / 0.01% / 0.62%
   * - Office_Products
     - user_split
     - 12,740 / 100.00% / 99.69%
     - 12,740 / 100.00% / 99.69%
     - 5,177 / 100.00% / 99.77%
     - 5,178 / 100.00% / 99.69%
     - 0.00% / 0.00% / 0.21%
   * - Office_Products
     - item_split
     - 12,740 / 100.00% / 99.69%
     - 10,829 / 100.00% / 99.70%
     - 637 / 100.00% / 99.69%
     - 1,274 / 100.00% / 99.69%
     - 0.00% / 0.00% / 0.23%
   * - Office_Products
     - leave_last_out
     - 12,740 / 100.00% / 99.69%
     - 12,738 / 100.00% / 99.69%
     - 12,194 / 100.00% / 99.69%
     - 11,905 / 100.00% / 99.69%
     - 0.00% / 0.00% / 0.21%
   * - Office_Products
     - temporal
     - 19,965 / 100.00% / 99.70%
     - 12,805 / 100.00% / 99.67%
     - 11,142 / 100.00% / 99.69%
     - 12,535 / 100.00% / 99.73%
     - 0.00% / <0.01% / 0.26%
   * - Patio_Lawn_and_Garden
     - user_split
     - 14,019 / 100.00% / 99.96%
     - 14,019 / 100.00% / 99.96%
     - 6,556 / 100.00% / 99.95%
     - 6,539 / 100.00% / 99.94%
     - 0.00% / 0.00% / 0.13%
   * - Patio_Lawn_and_Garden
     - item_split
     - 14,019 / 100.00% / 99.96%
     - 11,916 / 100.00% / 99.96%
     - 701 / 100.00% / 100.00%
     - 1,402 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.14%
   * - Patio_Lawn_and_Garden
     - leave_last_out
     - 14,019 / 100.00% / 99.96%
     - 14,019 / 100.00% / 99.96%
     - 13,333 / 100.00% / 99.96%
     - 12,826 / 100.00% / 99.96%
     - 0.00% / 0.00% / 0.14%
   * - Patio_Lawn_and_Garden
     - temporal
     - 14,329 / 100.00% / 99.97%
     - 8,247 / 100.00% / 99.95%
     - 10,667 / 100.00% / 99.95%
     - 9,718 / 100.00% / 99.95%
     - 0.00% / 0.00% / 0.12%
   * - Pet_Supplies
     - user_split
     - 19,880 / 100.00% / 99.98%
     - 19,880 / 100.00% / 99.98%
     - 8,436 / 100.00% / 99.98%
     - 8,533 / 100.00% / 100.00%
     - 0.00% / <0.01% / 0.21%
   * - Pet_Supplies
     - item_split
     - 19,880 / 100.00% / 99.98%
     - 16,898 / 100.00% / 99.98%
     - 994 / 100.00% / 100.00%
     - 1,988 / 100.00% / 100.00%
     - 0.00% / <0.01% / 0.21%
   * - Pet_Supplies
     - leave_last_out
     - 19,880 / 100.00% / 99.98%
     - 19,880 / 100.00% / 99.98%
     - 16,106 / 100.00% / 99.98%
     - 15,361 / 100.00% / 99.98%
     - 0.00% / <0.01% / 0.22%
   * - Pet_Supplies
     - temporal
     - 18,400 / 100.00% / 99.98%
     - 13,065 / 100.00% / 99.98%
     - 11,921 / 100.00% / 99.97%
     - 12,385 / 100.00% / 99.98%
     - 0.00% / <0.01% / 0.15%
   * - Software
     - user_split
     - 13,664 / 100.00% / 100.00%
     - 13,664 / 100.00% / 100.00%
     - 4,254 / 100.00% / 100.00%
     - 4,129 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.43%
   * - Software
     - item_split
     - 13,664 / 100.00% / 100.00%
     - 11,613 / 100.00% / 100.00%
     - 684 / 100.00% / 100.00%
     - 1,367 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.39%
   * - Software
     - leave_last_out
     - 13,664 / 100.00% / 100.00%
     - 13,639 / 100.00% / 100.00%
     - 9,800 / 100.00% / 100.00%
     - 9,400 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.42%
   * - Software
     - temporal
     - 14,548 / 100.00% / >99.99%
     - 11,639 / 100.00% / 100.00%
     - 5,902 / 100.00% / 100.00%
     - 3,762 / 100.00% / 100.00%
     - 0.00% / <0.01% / 0.71%
   * - Sports_and_Outdoors
     - user_split
     - 19,023 / 100.00% / 100.00%
     - 19,023 / 100.00% / 100.00%
     - 7,398 / 100.00% / 100.00%
     - 7,431 / 100.00% / 100.00%
     - 0.00% / 0.01% / 0.27%
   * - Sports_and_Outdoors
     - item_split
     - 19,023 / 100.00% / 100.00%
     - 16,168 / 100.00% / 100.00%
     - 952 / 100.00% / 100.00%
     - 1,903 / 100.00% / 100.00%
     - 0.00% / 0.01% / 0.29%
   * - Sports_and_Outdoors
     - leave_last_out
     - 19,023 / 100.00% / 100.00%
     - 19,023 / 100.00% / 100.00%
     - 17,601 / 100.00% / 100.00%
     - 16,816 / 100.00% / 100.00%
     - 0.00% / 0.01% / 0.30%
   * - Sports_and_Outdoors
     - temporal
     - 18,592 / 100.00% / 100.00%
     - 11,498 / 100.00% / 100.00%
     - 10,372 / 100.00% / 100.00%
     - 11,653 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.20%
   * - Subscription_Boxes
     - user_split
     - 291 / 99.66% / 100.00%
     - 291 / 99.66% / 100.00%
     - 28 / 100.00% / 100.00%
     - 36 / 97.22% / 100.00%
     - 0.19% / 0.19% / 6.12%
   * - Subscription_Boxes
     - item_split
     - 327 / 99.69% / 100.00%
     - 277 / 99.64% / 100.00%
     - 16 / 100.00% / 100.00%
     - 31 / 100.00% / 100.00%
     - 0.29% / 0.29% / 7.76%
   * - Subscription_Boxes
     - leave_last_out
     - 86 / 100.00% / 100.00%
     - 52 / 100.00% / 100.00%
     - 23 / 100.00% / 100.00%
     - 21 / 100.00% / 100.00%
     - 0.00% / 0.00% / 8.82%
   * - Subscription_Boxes
     - temporal
     - 191 / 100.00% / 100.00%
     - 86 / 100.00% / 100.00%
     - 62 / 100.00% / 100.00%
     - 34 / 100.00% / 100.00%
     - 0.00% / 0.00% / 4.51%
   * - Tools_and_Home_Improvement
     - user_split
     - 17,307 / 100.00% / >99.99%
     - 17,307 / 100.00% / >99.99%
     - 8,377 / 100.00% / 100.00%
     - 8,199 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.14%
   * - Tools_and_Home_Improvement
     - item_split
     - 17,307 / 100.00% / >99.99%
     - 14,710 / 100.00% / >99.99%
     - 866 / 100.00% / 100.00%
     - 1,731 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.11%
   * - Tools_and_Home_Improvement
     - leave_last_out
     - 17,307 / 100.00% / >99.99%
     - 17,307 / 100.00% / >99.99%
     - 15,754 / 100.00% / 100.00%
     - 15,151 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.15%
   * - Tools_and_Home_Improvement
     - temporal
     - 15,514 / 100.00% / 100.00%
     - 11,124 / 100.00% / 100.00%
     - 11,518 / 100.00% / 100.00%
     - 11,481 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.11%
   * - Toys_and_Games
     - user_split
     - 15,609 / 100.00% / >99.99%
     - 15,609 / 100.00% / >99.99%
     - 7,090 / 100.00% / 99.99%
     - 7,031 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.09%
   * - Toys_and_Games
     - item_split
     - 15,609 / 100.00% / >99.99%
     - 13,267 / 100.00% / >99.99%
     - 781 / 100.00% / 100.00%
     - 1,561 / 100.00% / 100.00%
     - 0.00% / 0.00% / 0.10%
   * - Toys_and_Games
     - leave_last_out
     - 15,609 / 100.00% / >99.99%
     - 15,608 / 100.00% / >99.99%
     - 14,818 / 100.00% / >99.99%
     - 14,342 / 100.00% / >99.99%
     - 0.00% / 0.00% / 0.10%
   * - Toys_and_Games
     - temporal
     - 17,369 / 100.00% / 99.99%
     - 10,382 / 100.00% / >99.99%
     - 8,553 / 100.00% / 99.98%
     - 11,428 / 100.00% / >99.99%
     - 0.00% / 0.00% / 0.05%
   * - Video_Games
     - user_split
     - 19,227 / 100.00% / >99.99%
     - 19,227 / 100.00% / >99.99%
     - 5,574 / 100.00% / 100.00%
     - 5,551 / 100.00% / 100.00%
     - 0.00% / 0.07% / 1.63%
   * - Video_Games
     - item_split
     - 19,227 / 100.00% / >99.99%
     - 16,342 / 100.00% / >99.99%
     - 962 / 100.00% / 100.00%
     - 1,923 / 100.00% / 100.00%
     - 0.00% / 0.08% / 1.72%
   * - Video_Games
     - leave_last_out
     - 19,227 / 100.00% / >99.99%
     - 19,202 / 100.00% / >99.99%
     - 15,526 / 100.00% / >99.99%
     - 15,042 / 100.00% / 100.00%
     - 0.00% / 0.07% / 1.68%
   * - Video_Games
     - temporal
     - 16,463 / 100.00% / >99.99%
     - 11,338 / 100.00% / >99.99%
     - 7,650 / 100.00% / 99.99%
     - 7,253 / 100.00% / 100.00%
     - 0.00% / 0.08% / 1.64%

Reproducing the audit
---------------------

``examples/validation/amazon_metadata_audit.py catalogs`` rebuilds the selected
split payloads from sealed prepared inputs using the original frozen
profiler/builder and checks their measured counts. It saves retained item IDs and
training item frequencies, not reusable checkpoints. Its ``scan`` mode streams
the cached metadata in bounded memory, deduplicates retained item IDs using the
adapter's first-record policy, and measures exactly those catalogs. It does not
download missing sources; a missing retained metadata record fails the audit.
Use separate output directories and keep the frozen audit source unchanged.
The published archive retains the original pre-change formatter fingerprints.
New scans also fingerprint the live adapter and base formatter, preventing a
resume from silently combining results produced by different implementations.
