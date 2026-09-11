Datasets
========

Available loaders, metadata, and measured defaults. See :ref:`cite-datasets`
for citations, :doc:`cli-reference` for parameters, and :doc:`examples` for code.

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

.. _dataset-default-table-guide:

Reading the default tables
--------------------------

* **Support:** minimum user/item interactions. Totals are after preprocessing;
  temporal totals span all stages. DBbook official totals use training data only.
* **Val/Test:** eligible users, distinct target items, and **cold %** (items absent
  from original training interactions). **†** means fewer than 1,000 users.
* **Metadata:** union of measured catalogs, including preprocessed items.
  Image URLs are not validated; text means constructed ``entity_text``.
  Feature coverage uses this union; ``d`` is vector dimension. Unknown is not zero.
* **Splits:** random item defaults are **85/5/10** train/val/test. Random splits
  deduplicate repeated user/item pairs; ordered splits retain events.

Defaults binarize retained feedback; measured does not mean paper-reproducing.
See :ref:`dataset-default-table-details` for definitions, methodology, and
the :download:`non-Amazon measurement archive <_static/dataset-default-measurements.json>`.

.. _dataset-ml1m:

MovieLens 1M
------------

**Loading:** ``dataset="ml1m"`` downloads GroupLens ratings and beeFormer
descriptions; SWAP features are optional.

**Terms:** `MovieLens research-use conditions <https://grouplens.org/datasets/movielens/1m/>`_;
commercial use requires permission. Cite the description/feature sources too.

**Defaults:** Positive ratings and support filtering retain useful histories;
the 30-word minimum selects descriptive metadata. The temporal default is
poorly sized here: only **91 training users**.

.. list-table:: Installed MovieLens 1M preprocessing and split defaults
   :header-rows: 1

   * - Dataset / preprocessing
     - User split
     - Item split
     - Leave-last-out
     - Temporal
     - Item metadata (union of measured splits)
   * - | MovieLens 1M
       | Rating ≥4
       | Min text: 30 words
       | Seed: 42
     - | 5/1
       | 6,033 users
       | 3,295 items
       | Train: 4,533 users
       | Val: 500 users †
       | 1,860 items / 0.00% cold
       | Test: 1,000 users
       | 2,256 items / 0.00% cold
     - | 5/1
       | 6,033 users
       | 3,295 items
       | Train: 6,033 users
       | Val: 5,338 users
       | 165 items / 100.00% cold
       | Test: 5,643 users
       | 330 items / 100.00% cold
     - | 5/1
       | 6,033 users
       | 3,295 items
       | Train: 6,033 users
       | Val: 6,033 users
       | 1,722 items / 0.29% cold
       | Test: 6,033 users
       | 1,758 items / 0.17% cold
     - | 5/1
       | 943 users
       | 3,117 items
       | Train: 91 users
       | Val: 771 users †
       | 2,439 items / 26.98% cold
       | Test: 435 users †
       | 2,033 items / 19.43% cold
       | Window: 8136 h
     - | Union: 3,295 items
       | Image URLs: not exposed
       | ≥10 words: 3,295 (100.00%)
       | Precomputed features:
       | text/minilm: 3,295 (100.00%), 384d
       | text/mpnet: 3,295 (100.00%), 768d
       | image/resnet152: 2,942 (89.29%), 1,000d
       | image/vgg: 2,942 (89.29%), 4,096d
       | image/vit_cls: 2,942 (89.29%), 768d
       | image/vit_avg: 2,942 (89.29%), 768d
       | audio/vggish: 3,171 (96.24%), 128d
       | audio/whisper: 3,171 (96.24%), 512d
       | video/i3d: 3,170 (96.21%), 2,048d
       | video/r2p1d: 3,171 (96.24%), 512d

.. _dataset-ml20m:

MovieLens 20M
-------------

**Loading:** ``dataset="ml20m"`` downloads GroupLens ratings and beeFormer
descriptions; user-tag annotations are optional.

**Terms:** `GroupLens usage conditions <https://grouplens.org/datasets/movielens/20m/>`_;
commercial use requires permission. Cite beeFormer when using its descriptions.

**Defaults:** MultVAE-style positive-feedback filtering, plus a 30-word text
minimum. Holdout sizes are practical defaults, not the paper's exact split.

.. list-table:: Installed MovieLens 20M preprocessing and split defaults
   :header-rows: 1

   * - Dataset / preprocessing
     - User split
     - Item split
     - Leave-last-out
     - Temporal
     - Item metadata (union of measured splits)
   * - | MovieLens 20M
       | Rating ≥4
       | Min text: 30 words
       | Seed: 42
     - | 5/1
       | 136,589 users
       | 16,902 items
       | Train: 129,089 users
       | Val: 2,500 users
       | 4,549 items / 0.00% cold
       | Test: 5,000 users
       | 5,972 items / 0.00% cold
     - | 5/1
       | 136,589 users
       | 16,902 items
       | Train: 136,589 users
       | Val: 102,285 users
       | 846 items / 100.00% cold
       | Test: 124,822 users
       | 1,691 items / 100.00% cold
     - | 5/1
       | 136,589 users
       | 16,902 items
       | Train: 136,589 users
       | Val: 136,589 users
       | 7,353 items / 0.60% cold
       | Test: 136,589 users
       | 7,784 items / 0.90% cold
     - | 5/1
       | 5,903 users
       | 15,399 items
       | Train: 3,801 users
       | Val: 3,323 users
       | 6,705 items / 12.17% cold
       | Test: 3,217 users
       | 7,346 items / 18.46% cold
       | Window: 8136 h
     - | Union: 16,902 items
       | Image URLs: not exposed
       | ≥10 words: 16,902 (100.00%)

.. _dataset-goodbooks:

Goodbooks-10k
-------------

**Loading:** ``dataset="goodbooks"`` downloads book ratings/metadata and
beeFormer descriptions; tag annotations are optional.

**Terms:** `CC BY-SA 4.0 <https://github.com/zygmuntz/goodbooks-10k/blob/master/LICENSE>`_
for the upstream release; cite the additional description source.

**Defaults:** Positive ratings, minimum interaction support, and 30-word metadata
give a text-ready benchmark. No timestamps: user/item splits only.

.. list-table:: Installed Goodbooks-10k preprocessing and split defaults
   :header-rows: 1

   * - Dataset / preprocessing
     - User split
     - Item split
     - Leave-last-out
     - Temporal
     - Item metadata (union of measured splits)
   * - | Goodbooks-10k
       | Rating ≥4
       | Min text: 30 words
       | Seed: 0
     - | 5/1
       | 53,365 users
       | 9,975 items
       | Train: 49,865 users
       | Val: 1,000 users
       | 5,361 items / 0.00% cold
       | Test: 2,500 users
       | 7,820 items / 0.00% cold
     - | 5/1
       | 53,365 users
       | 9,975 items
       | Train: 53,365 users
       | Val: 51,301 users
       | 499 items / 100.00% cold
       | Test: 53,070 users
       | 998 items / 100.00% cold
     - | Not supported
       | No interaction timestamps
     - | Not supported
       | No interaction timestamps
     - | Union: 9,975 items
       | Image URLs: not exposed
       | ≥10 words: 9,975 (100.00%)

.. _dataset-amazon2023:

Amazon Reviews 2023
-------------------

**Loading:** ``dataset="amazon2023"`` downloads one category's ratings and item
metadata, not review text. ``include_image_urls=True`` retains image links.

**Terms:** Follow the `Amazon Reviews 2023 release's usage conditions
<https://amazon-reviews-2023.github.io/>`_ and source attribution.

**Defaults:** All ratings count as feedback. Category-specific support filters
keep graphs manageable while preserving evaluation users; no sampling or text
minimum. Text fields also vary by category. Small holdouts remain flagged (†).

Details: :doc:`amazon-profiling`, :doc:`amazon-metadata`, and
:download:`measured profiles <_static/amazon-default-profiles.json>`.



.. list-table:: Installed category-specific Amazon support defaults
   :header-rows: 1

   * - Category
     - User split
     - Item split
     - Leave-last-out
     - Temporal
     - Item metadata (union of splits)
   * - All_Beauty
     - | 2/2
       | 22,634 users
       | 12,105 items
       | Val: 2,097 users
       | 1,826 items / 0.00% cold
       | Test: 2,068 users
       | 1,854 items / 0.00% cold
     - | 2/2
       | 22,634 users
       | 12,105 items
       | Val: 2,414 users
       | 602 items / 100.00% cold
       | Test: 4,813 users
       | 1,203 items / 100.00% cold
     - | 4/1
       | 2,909 users
       | 10,266 items
       | Val: 2,909 users
       | 2,503 items / 60.53% cold
       | Test: 2,909 users
       | 2,487 items / 64.41% cold
     - | 3/1
       | 3,622 users
       | 10,734 items
       | Val: 1,410 users
       | 1,657 items / 81.11% cold
       | Test: 670 users †
       | 618 items / 89.97% cold
     - | Union: 18,755 items
       | ≥1 image: 18,755 (100.00%)
       | ≥10 words: 18,528 (98.79%)
   * - Amazon_Fashion
     - | 2/4
       | 35,714 users
       | 10,217 items
       | Val: 3,551 users
       | 2,779 items / 0.00% cold
       | Test: 3,560 users
       | 2,692 items / 0.00% cold
     - | 2/4
       | 35,714 users
       | 10,217 items
       | Val: 3,762 users
       | 511 items / 100.00% cold
       | Test: 6,772 users
       | 1,022 items / 100.00% cold
     - | 4/2
       | 1,717 users
       | 4,098 items
       | Val: 1,717 users
       | 1,292 items / 15.48% cold
       | Test: 1,717 users
       | 1,190 items / 21.18% cold
     - | 2/2
       | 3,157 users
       | 4,080 items
       | Val: 1,116 users
       | 673 items / 72.07% cold
       | Test: 791 users †
       | 467 items / 80.09% cold
     - | Union: 13,201 items
       | ≥1 image: 13,201 (100.00%)
       | ≥10 words: 13,093 (99.18%)
   * - Appliances
     - | 3/2
       | 52,101 users
       | 19,327 items
       | Val: 4,872 users
       | 3,391 items / 0.00% cold
       | Test: 4,859 users
       | 3,277 items / 0.00% cold
     - | 3/2
       | 52,101 users
       | 19,327 items
       | Val: 9,980 users
       | 967 items / 100.00% cold
       | Test: 16,170 users
       | 1,933 items / 100.00% cold
     - | 4/2
       | 14,826 users
       | 10,896 items
       | Val: 14,826 users
       | 6,274 items / 14.87% cold
       | Test: 14,826 users
       | 6,215 items / 17.28% cold
     - | 2/2
       | 71,639 users
       | 16,820 items
       | Val: 28,766 users
       | 8,307 items / 37.52% cold
       | Test: 20,484 users
       | 7,199 items / 47.84% cold
     - | Union: 22,334 items
       | ≥1 image: 22,334 (100.00%)
       | ≥10 words: 22,323 (99.95%)
   * - Arts_Crafts_and_Sewing
     - | 5/16
       | 97,533 users
       | 18,043 items
       | Val: 5,000 users
       | 6,506 items / 0.00% cold
       | Test: 5,000 users
       | 6,613 items / 0.00% cold
     - | 5/16
       | 97,533 users
       | 18,043 items
       | Val: 31,814 users
       | 903 items / 100.00% cold
       | Test: 52,915 users
       | 1,805 items / 100.00% cold
     - | 5/16
       | 97,533 users
       | 18,043 items
       | Val: 97,533 users
       | 16,863 items / 0.00% cold
       | Test: 97,533 users
       | 16,459 items / 0.00% cold
     - | 3/12
       | 96,749 users
       | 12,713 items
       | Val: 42,979 users
       | 9,307 items / 21.85% cold
       | Test: 29,693 users
       | 9,221 items / 37.64% cold
     - | Union: 19,690 items
       | ≥1 image: 19,680 (99.95%)
       | ≥10 words: 19,688 (99.99%)
   * - Automotive
     - | 7/25
       | 99,264 users
       | 16,079 items
       | Val: 5,000 users
       | 7,492 items / 0.00% cold
       | Test: 5,000 users
       | 7,574 items / 0.00% cold
     - | 7/25
       | 99,264 users
       | 16,079 items
       | Val: 42,440 users
       | 804 items / 100.00% cold
       | Test: 66,549 users
       | 1,608 items / 100.00% cold
     - | 7/25
       | 99,264 users
       | 16,079 items
       | Val: 99,264 users
       | 15,101 items / 0.00% cold
       | Test: 99,264 users
       | 14,639 items / 0.00% cold
     - | 6/12
       | 86,556 users
       | 19,182 items
       | Val: 49,082 users
       | 15,181 items / 25.95% cold
       | Test: 30,194 users
       | 12,891 items / 29.71% cold
     - | Union: 20,809 items
       | ≥1 image: 20,808 (>99.99%)
       | ≥10 words: 20,808 (>99.99%)
   * - Baby_Products
     - | 6/9
       | 85,240 users
       | 17,987 items
       | Val: 5,000 users
       | 6,029 items / 0.00% cold
       | Test: 5,000 users
       | 6,056 items / 0.00% cold
     - | 6/9
       | 85,240 users
       | 17,987 items
       | Val: 29,585 users
       | 900 items / 100.00% cold
       | Test: 49,964 users
       | 1,799 items / 100.00% cold
     - | 6/9
       | 85,240 users
       | 17,987 items
       | Val: 85,240 users
       | 14,115 items / 0.03% cold
       | Test: 85,240 users
       | 13,560 items / 0.03% cold
     - | 3/7
       | 95,735 users
       | 13,365 items
       | Val: 38,591 users
       | 7,822 items / 22.00% cold
       | Test: 25,313 users
       | 8,198 items / 43.40% cold
     - | Union: 20,108 items
       | ≥1 image: 20,105 (99.99%)
       | ≥10 words: 20,103 (99.98%)
   * - Beauty_and_Personal_Care
     - | 8/25
       | 95,840 users
       | 16,817 items
       | Val: 5,000 users
       | 7,981 items / 0.00% cold
       | Test: 5,000 users
       | 8,008 items / 0.00% cold
     - | 8/25
       | 95,840 users
       | 16,817 items
       | Val: 44,765 users
       | 841 items / 100.00% cold
       | Test: 66,388 users
       | 1,682 items / 100.00% cold
     - | 8/25
       | 95,840 users
       | 16,817 items
       | Val: 95,840 users
       | 15,180 items / 0.00% cold
       | Test: 95,840 users
       | 14,750 items / 0.00% cold
     - | 6/18
       | 95,044 users
       | 14,656 items
       | Val: 50,440 users
       | 9,685 items / 28.90% cold
       | Test: 40,215 users
       | 11,789 items / 48.75% cold
     - | Union: 19,153 items
       | ≥1 image: 19,153 (100.00%)
       | ≥10 words: 19,153 (100.00%)
   * - Books
     - | 5/17
       | 428,352 users
       | 94,864 items
       | Val: 20,000 users
       | 31,567 items / 0.00% cold
       | Test: 20,000 users
       | 32,453 items / 0.00% cold
     - | 5/17
       | 428,352 users
       | 94,864 items
       | Val: 161,770 users
       | 4,744 items / 100.00% cold
       | Test: 241,076 users
       | 9,487 items / 100.00% cold
     - | 5/17
       | 428,352 users
       | 94,864 items
       | Val: 428,352 users
       | 80,327 items / 0.04% cold
       | Test: 428,352 users
       | 76,778 items / 0.05% cold
     - | 6/6
       | 112,433 users
       | 96,465 items
       | Val: 63,581 users
       | 37,923 items / 28.87% cold
       | Test: 41,812 users
       | 26,194 items / 40.21% cold
     - | Union: 127,961 items
       | ≥1 image: 113,452 (88.66%)
       | ≥10 words: 127,913 (99.96%)
   * - CDs_and_Vinyl
     - | 5/16
       | 66,625 users
       | 18,586 items
       | Val: 5,000 users
       | 8,295 items / 0.00% cold
       | Test: 5,000 users
       | 8,493 items / 0.00% cold
     - | 5/16
       | 66,625 users
       | 18,586 items
       | Val: 24,546 users
       | 930 items / 100.00% cold
       | Test: 38,473 users
       | 1,859 items / 100.00% cold
     - | 5/16
       | 66,625 users
       | 18,586 items
       | Val: 66,625 users
       | 16,271 items / <0.01% cold
       | Test: 66,625 users
       | 16,039 items / 0.01% cold
     - | 2/5
       | 29,237 users
       | 13,650 items
       | Val: 13,120 users
       | 6,986 items / 22.70% cold
       | Test: 6,481 users
       | 4,566 items / 24.11% cold
     - | Union: 23,072 items
       | ≥1 image: 23,054 (99.92%)
       | ≥10 words: 23,072 (100.00%)
   * - Cell_Phones_and_Accessories
     - | 6/13
       | 92,781 users
       | 19,322 items
       | Val: 5,000 users
       | 6,692 items / 0.00% cold
       | Test: 5,000 users
       | 6,736 items / 0.00% cold
     - | 6/13
       | 92,781 users
       | 19,322 items
       | Val: 28,616 users
       | 967 items / 100.00% cold
       | Test: 52,067 users
       | 1,933 items / 100.00% cold
     - | 6/13
       | 92,781 users
       | 19,322 items
       | Val: 92,781 users
       | 16,504 items / 0.04% cold
       | Test: 92,781 users
       | 15,069 items / 0.05% cold
     - | 5/8
       | 89,713 users
       | 19,096 items
       | Val: 41,928 users
       | 7,932 items / 38.87% cold
       | Test: 35,885 users
       | 10,338 items / 66.97% cold
     - | Union: 23,458 items
       | ≥1 image: 23,458 (100.00%)
       | ≥10 words: 23,454 (99.98%)
   * - Clothing_Shoes_and_Jewelry
     - | 10/35
       | 93,809 users
       | 14,289 items
       | Val: 5,000 users
       | 8,145 items / 0.00% cold
       | Test: 5,000 users
       | 8,166 items / 0.00% cold
     - | 10/35
       | 93,809 users
       | 14,289 items
       | Val: 46,057 users
       | 715 items / 100.00% cold
       | Test: 70,344 users
       | 1,429 items / 100.00% cold
     - | 10/35
       | 93,809 users
       | 14,289 items
       | Val: 93,809 users
       | 13,220 items / 0.00% cold
       | Test: 93,809 users
       | 12,877 items / 0.00% cold
     - | 10/17
       | 113,384 users
       | 24,824 items
       | Val: 76,251 users
       | 20,663 items / 54.84% cold
       | Test: 59,629 users
       | 19,421 items / 57.30% cold
     - | Union: 24,842 items
       | ≥1 image: 24,839 (99.99%)
       | ≥10 words: 24,842 (100.00%)
   * - Digital_Music
     - | 2/2
       | 2,782 users
       | 2,389 items
       | Val: 254 users †
       | 252 items / 0.00% cold
       | Test: 244 users †
       | 240 items / 0.00% cold
     - | 2/2
       | 2,782 users
       | 2,389 items
       | Val: 314 users †
       | 119 items / 100.00% cold
       | Test: 563 users †
       | 236 items / 100.00% cold
     - | 4/1
       | 2,243 users
       | 16,240 items
       | Val: 2,243 users
       | 2,184 items / 85.21% cold
       | Test: 2,243 users
       | 2,177 items / 86.08% cold
     - | 2/1
       | 2,115 users
       | 8,154 items
       | Val: 862 users †
       | 1,121 items / 92.51% cold
       | Test: 604 users †
       | 735 items / 93.61% cold
     - | Union: 18,995 items
       | ≥1 image: 18,995 (100.00%)
       | ≥10 words: 16,956 (89.27%)
   * - Electronics
     - | 8/16
       | 494,089 users
       | 92,180 items
       | Val: 20,000 users
       | 31,288 items / 0.00% cold
       | Test: 20,000 users
       | 31,489 items / 0.00% cold
     - | 8/16
       | 494,089 users
       | 92,180 items
       | Val: 224,865 users
       | 4,609 items / 100.00% cold
       | Test: 347,160 users
       | 9,218 items / 100.00% cold
     - | 8/16
       | 494,089 users
       | 92,180 items
       | Val: 494,089 users
       | 72,119 items / <0.01% cold
       | Test: 494,089 users
       | 66,509 items / <0.01% cold
     - | 6/12
       | 484,391 users
       | 78,239 items
       | Val: 252,303 users
       | 39,800 items / 21.04% cold
       | Test: 177,638 users
       | 40,647 items / 40.37% cold
     - | Union: 99,261 items
       | ≥1 image: 99,256 (>99.99%)
       | ≥10 words: 99,242 (99.98%)
   * - Gift_Cards
     - | 2/1
       | 11,426 users
       | 749 items
       | Val: 1,135 users
       | 230 items / 0.00% cold
       | Test: 1,131 users
       | 240 items / 0.00% cold
     - | 2/2
       | 11,333 users
       | 575 items
       | Val: 2,285 users
       | 29 items / 100.00% cold
       | Test: 1,445 users
       | 58 items / 100.00% cold
     - | 4/1
       | 1,201 users
       | 503 items
       | Val: 1,201 users
       | 253 items / 14.62% cold
       | Test: 1,201 users
       | 244 items / 14.75% cold
     - | 2/1
       | 2,313 users
       | 421 items
       | Val: 1,092 users
       | 123 items / 28.46% cold
       | Test: 577 users †
       | 120 items / 38.33% cold
     - | Union: 749 items
       | ≥1 image: 749 (100.00%)
       | ≥10 words: 748 (99.87%)
   * - Grocery_and_Gourmet_Food
     - | 7/24
       | 98,300 users
       | 16,735 items
       | Val: 5,000 users
       | 7,768 items / 0.00% cold
       | Test: 5,000 users
       | 7,813 items / 0.00% cold
     - | 7/24
       | 98,300 users
       | 16,735 items
       | Val: 42,173 users
       | 837 items / 100.00% cold
       | Test: 63,925 users
       | 1,674 items / 100.00% cold
     - | 7/24
       | 98,300 users
       | 16,735 items
       | Val: 98,300 users
       | 15,347 items / 0.00% cold
       | Test: 98,300 users
       | 14,851 items / 0.00% cold
     - | 6/13
       | 86,148 users
       | 19,238 items
       | Val: 50,360 users
       | 14,478 items / 32.93% cold
       | Test: 34,731 users
       | 13,429 items / 39.73% cold
     - | Union: 21,273 items
       | ≥1 image: 21,262 (99.95%)
       | ≥10 words: 21,266 (99.97%)
   * - Handmade_Products
     - | 2/2
       | 22,156 users
       | 11,878 items
       | Val: 2,006 users
       | 1,725 items / 0.00% cold
       | Test: 2,040 users
       | 1,788 items / 0.00% cold
     - | 2/2
       | 22,156 users
       | 11,878 items
       | Val: 2,335 users
       | 587 items / 100.00% cold
       | Test: 4,283 users
       | 1,180 items / 100.00% cold
     - | 4/1
       | 3,716 users
       | 14,723 items
       | Val: 3,716 users
       | 3,350 items / 76.24% cold
       | Test: 3,716 users
       | 3,347 items / 77.14% cold
     - | 3/1
       | 3,774 users
       | 11,710 items
       | Val: 1,410 users
       | 2,029 items / 88.66% cold
       | Test: 1,251 users
       | 1,708 items / 92.62% cold
     - | Union: 23,704 items
       | ≥1 image: 23,704 (100.00%)
       | ≥10 words: 23,703 (>99.99%)
   * - Health_and_Household
     - | 10/20
       | 84,204 users
       | 19,922 items
       | Val: 5,000 users
       | 9,180 items / 0.00% cold
       | Test: 5,000 users
       | 8,986 items / 0.00% cold
     - | 10/20
       | 84,204 users
       | 19,922 items
       | Val: 44,334 users
       | 997 items / 100.00% cold
       | Test: 63,720 users
       | 1,993 items / 100.00% cold
     - | 10/20
       | 84,204 users
       | 19,922 items
       | Val: 84,204 users
       | 16,439 items / 0.00% cold
       | Test: 84,204 users
       | 15,703 items / 0.00% cold
     - | 8/15
       | 88,598 users
       | 19,907 items
       | Val: 50,052 users
       | 13,166 items / 26.50% cold
       | Test: 38,620 users
       | 14,545 items / 44.23% cold
     - | Union: 23,856 items
       | ≥1 image: 23,856 (100.00%)
       | ≥10 words: 23,856 (100.00%)
   * - Health_and_Personal_Care
     - | 2/1
       | 18,396 users
       | 17,288 items
       | Val: 1,041 users
       | 940 items / 0.00% cold
       | Test: 1,038 users
       | 877 items / 0.00% cold
     - | 2/1
       | 18,396 users
       | 17,288 items
       | Val: 1,758 users
       | 784 items / 100.00% cold
       | Test: 3,890 users
       | 1,585 items / 100.00% cold
     - | 4/1
       | 1,109 users
       | 3,448 items
       | Val: 1,109 users
       | 911 items / 42.26% cold
       | Test: 1,109 users
       | 857 items / 47.37% cold
     - | 2/1
       | 5,291 users
       | 7,604 items
       | Val: 2,100 users
       | 1,603 items / 72.99% cold
       | Test: 1,040 users
       | 762 items / 77.30% cold
     - | Union: 17,288 items
       | ≥1 image: 17,288 (100.00%)
       | ≥10 words: 16,849 (97.46%)
   * - Home_and_Kitchen
     - | 12/35
       | 96,909 users
       | 17,431 items
       | Val: 5,000 users
       | 9,721 items / 0.00% cold
       | Test: 5,000 users
       | 9,750 items / 0.00% cold
     - | 12/35
       | 96,909 users
       | 17,431 items
       | Val: 51,957 users
       | 872 items / 100.00% cold
       | Test: 78,055 users
       | 1,744 items / 100.00% cold
     - | 12/35
       | 96,909 users
       | 17,431 items
       | Val: 96,909 users
       | 15,181 items / 0.00% cold
       | Test: 96,909 users
       | 14,374 items / 0.00% cold
     - | 10/25
       | 93,903 users
       | 14,879 items
       | Val: 53,997 users
       | 11,264 items / 20.23% cold
       | Test: 38,614 users
       | 10,726 items / 28.82% cold
     - | Union: 18,990 items
       | ≥1 image: 18,990 (100.00%)
       | ≥10 words: 18,990 (100.00%)
   * - Industrial_and_Scientific
     - | 4/8
       | 74,487 users
       | 19,251 items
       | Val: 5,000 users
       | 5,348 items / 0.00% cold
       | Test: 5,000 users
       | 5,350 items / 0.00% cold
     - | 4/8
       | 74,487 users
       | 19,251 items
       | Val: 17,942 users
       | 963 items / 100.00% cold
       | Test: 33,333 users
       | 1,926 items / 100.00% cold
     - | 4/8
       | 74,487 users
       | 19,251 items
       | Val: 74,487 users
       | 16,742 items / 0.12% cold
       | Test: 74,487 users
       | 16,000 items / 0.14% cold
     - | 3/5
       | 66,901 users
       | 17,690 items
       | Val: 31,809 users
       | 10,279 items / 36.56% cold
       | Test: 23,291 users
       | 10,034 items / 52.33% cold
     - | Union: 22,281 items
       | ≥1 image: 22,281 (100.00%)
       | ≥10 words: 22,278 (99.99%)
   * - Kindle_Store
     - | 12/76
       | 98,841 users
       | 17,800 items
       | Val: 5,000 users
       | 13,209 items / 0.00% cold
       | Test: 5,000 users
       | 13,406 items / 0.00% cold
     - | 12/76
       | 98,841 users
       | 17,800 items
       | Val: 63,226 users
       | 890 items / 100.00% cold
       | Test: 85,505 users
       | 1,780 items / 100.00% cold
     - | 12/76
       | 98,841 users
       | 17,800 items
       | Val: 98,841 users
       | 15,961 items / 0.00% cold
       | Test: 98,841 users
       | 15,154 items / 0.00% cold
     - | 8/50
       | 96,853 users
       | 19,957 items
       | Val: 60,020 users
       | 15,954 items / 16.75% cold
       | Test: 42,975 users
       | 14,155 items / 24.56% cold
     - | Union: 23,549 items
       | ≥1 image: 21,033 (89.32%)
       | ≥10 words: 23,549 (100.00%)
   * - Magazine_Subscriptions
     - | 2/1
       | 6,808 users
       | 2,003 items
       | Val: 1,115 users
       | 472 items / 0.00% cold
       | Test: 1,093 users
       | 463 items / 0.00% cold
     - | 2/2
       | 6,408 users
       | 1,241 items
       | Val: 1,070 users
       | 63 items / 100.00% cold
       | Test: 1,594 users
       | 124 items / 100.00% cold
     - | 4/1
       | 797 users
       | 1,110 items
       | Val: 797 users †
       | 353 items / 25.21% cold
       | Test: 797 users †
       | 381 items / 30.45% cold
     - | 2/1
       | 458 users
       | 522 items
       | Val: 151 users †
       | 117 items / 45.30% cold
       | Test: 74 users †
       | 66 items / 33.33% cold
     - | Union: 2,003 items
       | ≥1 image: 2,003 (100.00%)
       | ≥10 words: 1,870 (93.36%)
   * - Movies_and_TV
     - | 10/36
       | 97,871 users
       | 18,324 items
       | Val: 5,000 users
       | 10,254 items / 0.00% cold
       | Test: 5,000 users
       | 10,070 items / 0.00% cold
     - | 10/36
       | 97,871 users
       | 18,324 items
       | Val: 50,964 users
       | 917 items / 100.00% cold
       | Test: 76,392 users
       | 1,833 items / 100.00% cold
     - | 10/36
       | 97,871 users
       | 18,324 items
       | Val: 97,871 users
       | 16,454 items / 0.00% cold
       | Test: 97,871 users
       | 15,927 items / 0.00% cold
     - | 5/12
       | 60,122 users
       | 17,522 items
       | Val: 31,406 users
       | 10,612 items / 18.37% cold
       | Test: 16,203 users
       | 6,941 items / 23.51% cold
     - | Union: 24,791 items
       | ≥1 image: 24,791 (100.00%)
       | ≥10 words: 24,636 (99.37%)
   * - Musical_Instruments
     - | 5/7
       | 50,431 users
       | 16,766 items
       | Val: 5,000 users
       | 5,933 items / 0.00% cold
       | Test: 5,000 users
       | 5,863 items / 0.00% cold
     - | 5/7
       | 50,431 users
       | 16,766 items
       | Val: 18,353 users
       | 839 items / 100.00% cold
       | Test: 27,033 users
       | 1,677 items / 100.00% cold
     - | 5/7
       | 50,431 users
       | 16,766 items
       | Val: 50,431 users
       | 12,985 items / 0.05% cold
       | Test: 50,431 users
       | 12,681 items / 0.05% cold
     - | 2/5
       | 77,684 users
       | 15,914 items
       | Val: 32,616 users
       | 10,123 items / 23.50% cold
       | Test: 23,272 users
       | 9,093 items / 33.53% cold
     - | Union: 20,065 items
       | ≥1 image: 20,061 (99.98%)
       | ≥10 words: 20,057 (99.96%)
   * - Office_Products
     - | 5/18
       | 98,570 users
       | 12,740 items
       | Val: 5,000 users
       | 5,177 items / 0.00% cold
       | Test: 5,000 users
       | 5,178 items / 0.00% cold
     - | 5/18
       | 98,570 users
       | 12,740 items
       | Val: 30,907 users
       | 637 items / 100.00% cold
       | Test: 49,343 users
       | 1,274 items / 100.00% cold
     - | 5/18
       | 98,570 users
       | 12,740 items
       | Val: 98,570 users
       | 12,194 items / 0.02% cold
       | Test: 98,570 users
       | 11,905 items / 0.02% cold
     - | 4/8
       | 90,062 users
       | 19,965 items
       | Val: 39,715 users
       | 11,142 items / 22.83% cold
       | Test: 29,909 users
       | 12,535 items / 47.53% cold
     - | Union: 22,096 items
       | ≥1 image: 22,029 (99.70%)
       | ≥10 words: 22,095 (>99.99%)
   * - Patio_Lawn_and_Garden
     - | 6/21
       | 98,613 users
       | 14,019 items
       | Val: 5,000 users
       | 6,556 items / 0.00% cold
       | Test: 5,000 users
       | 6,539 items / 0.00% cold
     - | 6/21
       | 98,613 users
       | 14,019 items
       | Val: 33,334 users
       | 701 items / 100.00% cold
       | Test: 55,849 users
       | 1,402 items / 100.00% cold
     - | 6/21
       | 98,613 users
       | 14,019 items
       | Val: 98,613 users
       | 13,333 items / 0.00% cold
       | Test: 98,613 users
       | 12,826 items / 0.00% cold
     - | 5/13
       | 89,415 users
       | 14,329 items
       | Val: 52,279 users
       | 10,667 items / 35.60% cold
       | Test: 28,043 users
       | 9,718 items / 46.72% cold
     - | Union: 17,357 items
       | ≥1 image: 17,352 (99.97%)
       | ≥10 words: 17,357 (100.00%)
   * - Pet_Supplies
     - | 10/16
       | 89,150 users
       | 19,880 items
       | Val: 5,000 users
       | 8,436 items / 0.00% cold
       | Test: 5,000 users
       | 8,533 items / 0.00% cold
     - | 10/16
       | 89,150 users
       | 19,880 items
       | Val: 46,784 users
       | 994 items / 100.00% cold
       | Test: 68,722 users
       | 1,988 items / 100.00% cold
     - | 10/16
       | 89,150 users
       | 19,880 items
       | Val: 89,150 users
       | 16,106 items / 0.00% cold
       | Test: 89,150 users
       | 15,361 items / 0.00% cold
     - | 8/12
       | 88,279 users
       | 18,400 items
       | Val: 47,781 users
       | 11,921 items / 19.44% cold
       | Test: 34,883 users
       | 12,385 items / 36.15% cold
     - | Union: 22,828 items
       | ≥1 image: 22,824 (99.98%)
       | ≥10 words: 22,825 (99.99%)
   * - Software
     - | 6/6
       | 98,634 users
       | 13,664 items
       | Val: 5,000 users
       | 4,254 items / 0.00% cold
       | Test: 5,000 users
       | 4,129 items / 0.00% cold
     - | 6/6
       | 98,634 users
       | 13,664 items
       | Val: 34,962 users
       | 684 items / 100.00% cold
       | Test: 62,676 users
       | 1,367 items / 100.00% cold
     - | 6/6
       | 98,634 users
       | 13,664 items
       | Val: 98,634 users
       | 9,800 items / 0.17% cold
       | Test: 98,634 users
       | 9,400 items / 0.27% cold
     - | 2/2
       | 76,101 users
       | 14,548 items
       | Val: 29,981 users
       | 5,902 items / 22.89% cold
       | Test: 14,134 users
       | 3,762 items / 24.83% cold
     - | Union: 18,322 items
       | ≥1 image: 18,321 (>99.99%)
       | ≥10 words: 18,318 (99.98%)
   * - Sports_and_Outdoors
     - | 6/17
       | 99,084 users
       | 19,023 items
       | Val: 5,000 users
       | 7,398 items / 0.00% cold
       | Test: 5,000 users
       | 7,431 items / 0.00% cold
     - | 6/17
       | 99,084 users
       | 19,023 items
       | Val: 35,955 users
       | 952 items / 100.00% cold
       | Test: 55,955 users
       | 1,903 items / 100.00% cold
     - | 6/17
       | 99,084 users
       | 19,023 items
       | Val: 99,084 users
       | 17,601 items / 0.00% cold
       | Test: 99,084 users
       | 16,816 items / 0.00% cold
     - | 5/9
       | 61,892 users
       | 18,592 items
       | Val: 29,250 users
       | 10,372 items / 24.21% cold
       | Test: 20,849 users
       | 11,653 items / 48.85% cold
     - | Union: 25,641 items
       | ≥1 image: 25,641 (100.00%)
       | ≥10 words: 25,637 (99.98%)
   * - Subscription_Boxes
     - | 2/1
       | 572 users
       | 327 items
       | Val: 42 users †
       | 28 items / 0.00% cold
       | Test: 47 users †
       | 36 items / 0.00% cold
     - | 2/1
       | 572 users
       | 327 items
       | Val: 56 users †
       | 16 items / 100.00% cold
       | Test: 141 users †
       | 31 items / 100.00% cold
     - | 4/1
       | 24 users
       | 86 items
       | Val: 24 users †
       | 23 items / 78.26% cold
       | Test: 24 users †
       | 21 items / 95.24% cold
     - | 2/1
       | 152 users
       | 191 items
       | Val: 59 users †
       | 62 items / 77.42% cold
       | Test: 42 users †
       | 34 items / 88.24% cold
     - | Union: 327 items
       | ≥1 image: 327 (100.00%)
       | ≥10 words: 326 (99.69%)
   * - Tools_and_Home_Improvement
     - | 8/26
       | 94,834 users
       | 17,307 items
       | Val: 5,000 users
       | 8,377 items / 0.00% cold
       | Test: 5,000 users
       | 8,199 items / 0.00% cold
     - | 8/26
       | 94,834 users
       | 17,307 items
       | Val: 45,023 users
       | 866 items / 100.00% cold
       | Test: 66,710 users
       | 1,731 items / 100.00% cold
     - | 8/26
       | 94,834 users
       | 17,307 items
       | Val: 94,834 users
       | 15,754 items / 0.00% cold
       | Test: 94,834 users
       | 15,151 items / 0.00% cold
     - | 6/18
       | 99,815 users
       | 15,514 items
       | Val: 52,511 users
       | 11,518 items / 21.32% cold
       | Test: 37,242 users
       | 11,481 items / 33.10% cold
     - | Union: 19,936 items
       | ≥1 image: 19,935 (>99.99%)
       | ≥10 words: 19,936 (100.00%)
   * - Toys_and_Games
     - | 6/22
       | 98,065 users
       | 15,609 items
       | Val: 5,000 users
       | 7,090 items / 0.00% cold
       | Test: 5,000 users
       | 7,031 items / 0.00% cold
     - | 6/22
       | 98,065 users
       | 15,609 items
       | Val: 34,026 users
       | 781 items / 100.00% cold
       | Test: 56,767 users
       | 1,561 items / 100.00% cold
     - | 6/22
       | 98,065 users
       | 15,609 items
       | Val: 98,065 users
       | 14,818 items / <0.01% cold
       | Test: 98,065 users
       | 14,342 items / <0.01% cold
     - | 5/10
       | 55,927 users
       | 17,369 items
       | Val: 23,364 users
       | 8,553 items / 20.29% cold
       | Test: 20,412 users
       | 11,428 items / 54.60% cold
     - | Union: 23,646 items
       | ≥1 image: 23,644 (>99.99%)
       | ≥10 words: 23,644 (>99.99%)
   * - Video_Games
     - | 5/7
       | 88,081 users
       | 19,227 items
       | Val: 5,000 users
       | 5,574 items / 0.00% cold
       | Test: 5,000 users
       | 5,551 items / 0.00% cold
     - | 5/7
       | 88,081 users
       | 19,227 items
       | Val: 30,322 users
       | 962 items / 100.00% cold
       | Test: 48,790 users
       | 1,923 items / 100.00% cold
     - | 5/7
       | 88,081 users
       | 19,227 items
       | Val: 88,081 users
       | 15,526 items / 0.15% cold
       | Test: 88,081 users
       | 15,042 items / 0.17% cold
     - | 3/4
       | 70,806 users
       | 16,463 items
       | Val: 30,103 users
       | 7,650 items / 29.50% cold
       | Test: 21,917 users
       | 7,253 items / 48.01% cold
     - | Union: 23,083 items
       | ≥1 image: 23,082 (>99.99%)
       | ≥10 words: 23,057 (99.89%)

.. _dataset-steam:

Steam
-----

**Loading:** ``dataset="steam"`` downloads McAuley's review and game-metadata
archives, then caches parsed interactions as Parquet.

**Terms:** Retain `upstream usage and citation information
<https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data>`_.

**Defaults:** Every review, including negative ones, counts as feedback
(BERT4Rec's convention). Support filtering removes short histories; no text
minimum, so missing metadata does not erase interactions.

.. list-table:: Installed Steam preprocessing and split defaults
   :header-rows: 1

   * - Dataset / preprocessing
     - User split
     - Item split
     - Leave-last-out
     - Temporal
     - Item metadata (union of measured splits)
   * - | Steam
       | All feedback
       | Min text: 0 words
       | Seed: 42
     - | 5/1
       | 281,645 users
       | 15,051 items
       | Train: 261,645 users
       | Val: 9,994 users
       | 4,949 items / 0.00% cold
       | Test: 9,998 users
       | 4,897 items / 0.00% cold
     - | 5/1
       | 281,645 users
       | 15,051 items
       | Train: 281,639 users
       | Val: 98,181 users
       | 753 items / 100.00% cold
       | Test: 162,161 users
       | 1,506 items / 100.00% cold
     - | 5/1
       | 334,730 users
       | 15,068 items
       | Train: 334,730 users
       | Val: 334,730 users
       | 10,453 items / 2.06% cold
       | Test: 334,730 users
       | 10,621 items / 3.29% cold
     - | 5/1
       | 250,498 users
       | 14,965 items
       | Train: 91,041 users
       | Val: 138,434 users
       | 9,904 items / 35.00% cold
       | Test: 192,164 users
       | 13,781 items / 54.84% cold
       | Window: 8136 h
     - | Union: 15,068 items
       | Image URLs: not exposed
       | ≥10 words: 9,727 (64.55%)

.. _dataset-netflix:

Netflix Prize
-------------

**Loading:** ``dataset="netflix"`` reads the original-format ratings archive
from Internet Archive, including dates and movie titles/years.

**Terms:** Original Netflix Prize conditions apply; the
`archive mirror <https://archive.org/details/nf_prize_dataset.tar>`_ does not
replace those terms.

**Defaults:** MultVAE-style positive-rating and support filtering.
No text minimum: short movie titles remain usable metadata.

.. list-table:: Installed Netflix Prize preprocessing and split defaults
   :header-rows: 1

   * - Dataset / preprocessing
     - User split
     - Item split
     - Leave-last-out
     - Temporal
     - Item metadata (union of measured splits)
   * - | Netflix Prize
       | Rating ≥4
       | Min text: 0 words
       | Seed: 98765
     - | 5/1
       | 463,435 users
       | 17,769 items
       | Train: 383,435 users
       | Val: 40,000 users
       | 14,546 items / 0.00% cold
       | Test: 40,000 users
       | 14,474 items / 0.00% cold
     - | 5/1
       | 463,435 users
       | 17,769 items
       | Train: 463,434 users
       | Val: 377,662 users
       | 889 items / 100.00% cold
       | Test: 440,315 users
       | 1,777 items / 100.00% cold
     - | 5/1
       | 463,435 users
       | 17,769 items
       | Train: 463,435 users
       | Val: 463,435 users
       | 13,185 items / 0.00% cold
       | Test: 463,435 users
       | 13,433 items / 0.00% cold
     - | 5/1
       | 262,636 users
       | 17,766 items
       | Train: 58,419 users
       | Val: 130,014 users
       | 15,317 items / 22.21% cold
       | Test: 241,687 users
       | 17,612 items / 31.77% cold
       | Window: 8136 h
     - | Union: 17,769 items
       | Image URLs: not exposed
       | ≥10 words: 581 (3.27%)

.. _dataset-taste-profile:

MSD Taste Profile
-----------------

**Loading:** ``dataset="taste-profile"`` downloads the original user/song
play-count ZIP, not the full MSD audio features or metadata.

**Terms:** `MSD / Echo Nest usage conditions
<http://millionsongdataset.com/tasteprofile/>`_ apply.

**Defaults:** MultVAE-style support (20 interactions per user, 200 per item)
removes sparse tails; play counts become binary feedback. No timestamps or text.

.. list-table:: Installed MSD Taste Profile preprocessing and split defaults
   :header-rows: 1

   * - Dataset / preprocessing
     - User split
     - Item split
     - Leave-last-out
     - Temporal
     - Item metadata (union of measured splits)
   * - | MSD Taste Profile
       | All feedback
       | Min text: 0 words
       | Seed: 98765
     - | 20/200
       | 558,350 users
       | 36,385 items
       | Train: 458,350 users
       | Val: 50,000 users
       | 36,262 items / 0.00% cold
       | Test: 50,000 users
       | 36,256 items / 0.00% cold
     - | 20/200
       | 558,350 users
       | 36,385 items
       | Train: 558,350 users
       | Val: 478,324 users
       | 1,820 items / 100.00% cold
       | Test: 543,782 users
       | 3,639 items / 100.00% cold
     - | Not supported
       | No interaction timestamps
     - | Not supported
       | No interaction timestamps
     - | Union: 36,385 items
       | Image URLs: not exposed
       | ≥10 words: 0 (0.00%)

.. _dataset-gowalla:

Gowalla
-------

**Loading:** ``dataset="gowalla"`` downloads raw SNAP check-ins and coordinates,
not the social graph or pre-made LightGCN splits.

**Terms:** Retain `SNAP's source citation and usage information
<https://snap.stanford.edu/data/loc-gowalla.html>`_.

**Defaults:** NGCF-style 10/10 support. The **8,136-hour temporal default fails**
because its windows exceed the data span; :doc:`dataset-sweep` uses 720 hours.

.. list-table:: Installed Gowalla preprocessing and split defaults
   :header-rows: 1

   * - Dataset / preprocessing
     - User split
     - Item split
     - Leave-last-out
     - Temporal
     - Item metadata (union of measured splits)
   * - | Gowalla
       | All feedback
       | Min text: 0 words
       | Seed: 42
     - | 10/10
       | 29,858 users
       | 40,988 items
       | Train: 9,858 users
       | Val: 9,966 users
       | 28,823 items / 0.00% cold
       | Test: 9,961 users
       | 28,653 items / 0.00% cold
     - | 10/10
       | 29,858 users
       | 40,988 items
       | Train: 29,858 users
       | Val: 19,376 users
       | 2,050 items / 100.00% cold
       | Test: 25,848 users
       | 4,099 items / 100.00% cold
     - | 10/10
       | 52,985 users
       | 121,866 items
       | Train: 52,985 users
       | Val: 52,985 users
       | 36,247 items / 0.00% cold
       | Test: 52,985 users
       | 36,292 items / 0.00% cold
     - | 10/10
       | Build failed
       | ValueError: temporal_period_hours requires three target
       | windows shorter than the available 15024.074-hour timestamp
       | span
       | Window: 8136 h
     - | Union: 121,866 items
       | Image URLs: not exposed
       | ≥10 words: 0 (0.00%)

.. _dataset-dbbook:

DBbook
------

**Loading:** ``dataset="dbbook"`` downloads SWAP's reconstructed interaction
ZIP and DBpedia title mappings; abstract/cover embeddings are optional.

**Terms:** Consult the `SWAP release <https://zenodo.org/records/15403972>`_;
original DBbook/DBpedia rights still apply.

**Defaults:** Keep positive feedback and sufficiently supported histories,
without filtering short titles. ``official`` preserves the supplied test
boundary; see :ref:`dbbook-official-protocol` for validation details.

.. list-table:: Installed DBbook preprocessing and split defaults
   :header-rows: 1

   * - Dataset / preprocessing
     - User split
     - Item split
     - Leave-last-out
     - Temporal
     - Official
     - Item metadata (union of measured splits)
   * - | DBbook
       | Rating ≥1
       | Min text: 0 words
       | Seed: 42
     - | 5/1
       | 5,884 users
       | 6,538 items
       | Train: 4,384 users
       | Val: 496 users †
       | 916 items / 0.00% cold
       | Test: 998 users †
       | 1,443 items / 0.00% cold
     - | 5/1
       | 5,884 users
       | 6,538 items
       | Train: 5,883 users
       | Val: 2,498 users
       | 327 items / 100.00% cold
       | Test: 3,974 users
       | 654 items / 100.00% cold
     - | Not supported
       | No interaction timestamps
     - | Not supported
       | No interaction timestamps
     - | 5/1
       | 2,799 users
       | 4,548 items
       | Train: 2,799 users
       | Val: 2,799 users
       | 2,335 items / 20.00% cold
       | Test: 2,144 users
       | 2,742 items / 6.78% cold
       | Users/items above: training-only preprocessing
     - | Union: 6,538 items
       | Image URLs: not exposed
       | ≥10 words: 64 (0.98%)
       | Precomputed features:
       | text/minilm: 5,698 (87.15%), 384d
       | text/mpnet: 5,698 (87.15%), 768d
       | image/resnet152: 6,287 (96.16%), 1,000d
       | image/vgg: 6,287 (96.16%), 4,096d
       | image/vit_cls: 6,287 (96.16%), 768d
       | image/vit_avg: 6,287 (96.16%), 768d

.. _dataset-lfm2k:

Last.fm-2K
----------

**Loading:** ``dataset="lfm2k"`` loads HetRec artist listening counts and metadata
from SWAP's ZIP; optional media vectors are pooled by artist.

**Terms:** `Last.fm / HetRec non-commercial-use conditions
<https://grouplens.org/datasets/hetrec-2011/>`_; commercial use requires permission.
Cite SWAP for its features.

**Defaults:** Binarize play counts, filter short histories, and retain short
artist names. Tag-derived embeddings are **interaction-derived**, not independent
content metadata.

.. list-table:: Installed Last.fm-2K preprocessing and split defaults
   :header-rows: 1

   * - Dataset / preprocessing
     - User split
     - Item split
     - Leave-last-out
     - Temporal
     - Item metadata (union of measured splits)
   * - | Last.fm-2K
       | All feedback
       | Min text: 0 words
       | Seed: 42
     - | 5/1
       | 1,877 users
       | 17,617 items
       | Train: 1,277 users
       | Val: 200 users †
       | 1,070 items / 0.00% cold
       | Test: 400 users †
       | 1,731 items / 0.00% cold
     - | 5/1
       | 1,877 users
       | 17,617 items
       | Train: 1,877 users
       | Val: 1,748 users
       | 881 items / 100.00% cold
       | Test: 1,864 users
       | 1,762 items / 100.00% cold
     - | Not supported
       | No interaction timestamps
     - | Not supported
       | No interaction timestamps
     - | Union: 17,617 items
       | ≥1 image: 17,173 (97.48%)
       | ≥10 words: 19 (0.11%)
       | Precomputed features:
       | text/minilm: 12,124 (68.82%), 384d
       | text/mpnet: 12,124 (68.82%), 768d
       | image/resnet152: 2,314 (13.14%), 1,000d
       | image/vgg: 2,314 (13.14%), 4,096d
       | image/vit_cls: 2,314 (13.14%), 768d
       | image/vit_avg: 2,314 (13.14%), 768d
       | audio/vggish: 2,364 (13.42%), 128d
       | audio/whisper: 2,364 (13.42%), 512d
       | Text: interaction-derived tags
       | Media: pooled by artist ID

.. _dataset-item-embeddings:

Using item embeddings
---------------------

SWAP JSON features are opt-in, checksum-verified imports; they do not change
the interaction split. Other datasets use user-computed vectors in the same
checkpoint format. See :ref:`item-embedding-recipes` for import, storage, and
missing-feature handling; original dataset restrictions still apply.

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
