.. Licence cells verified against the upstream pages on 2026-10-24.

Dataset licences and terms of use
=================================

**Compresso Recsys downloads and caches; it never redistributes.** Every loader
fetches the data from its publisher, or asks you to fetch it yourself where the
publisher requires an account, and keeps the archive and a parsed cache in your
local data directory. The package ships no data. The project does not publish
checkpoints, model files, or benchmark artefacts derived from these datasets, and
it does not grant you any right to do so. The publisher's terms apply to you
directly.

Checkpoints built from a dataset contain its interaction data and are derived
works of it. **Sharing a checkpoint is redistribution of the data.** Before
sharing one, or anything trained on it, check the terms below. Rows marked ⚠
restrict or do not permit that, or their governing terms cannot be read.

The table summarises the terms so you can check a dataset is usable for your
purpose *before* you download it. It is not legal advice; the linked upstream
page is authoritative.

All pages were accessed on **2026-10-24**. "(via mirror)" marks rows whose
publisher page refused automated access; their terms were read from a dated Web
Archive capture or a verbatim README copy.

.. list-table::
   :header-rows: 1
   :widths: 13 19 22 9 12 9 16

   * - Dataset (key)
     - Upstream source
     - Licence / terms
     - Commercial use
     - Redistribution (incl. checkpoints)
     - Download
     - Citation
   * - ⚠ MovieLens 1M (``ml1m``) (via mirror)
     - `GroupLens <https://grouplens.org/datasets/movielens/1m/>`_
     - GroupLens README usage licence: research only; no endorsement; acknowledge in publications
     - No, without permission
     - **No**, without separate permission
     - automatic
     - Harper & Konstan, ACM TiiS 2015 (doi:10.1145/2827872)
   * - ⚠ MovieLens 20M (``ml20m``) (via mirror)
     - `GroupLens <https://grouplens.org/datasets/movielens/20m/>`_
     - Same GroupLens usage licence (same README text; not re-read for 20M)
     - No, without permission
     - **No**, without separate permission
     - automatic
     - Harper & Konstan 2015
   * - Goodbooks-10k (``goodbooks``)
     - `GitHub <https://github.com/zygmuntz/goodbooks-10k>`_
     - CC BY-SA 4.0
     - Yes
     - Yes, with attribution; **share-alike**: published checkpoints must be CC BY-SA 4.0
     - automatic
     - the repository
   * - ⚠ Amazon Reviews 2023 (``amazon2023``)
     - `McAuley Lab <https://amazon-reviews-2023.github.io/>`_ (Hugging Face or mirror)
     - **No licence stated** (citation requested); reviews are Amazon content, research use by convention
     - Not granted
     - **Not granted**
     - automatic
     - Hou et al., arXiv:2403.03952 (2024)
   * - ⚠ Steam (``steam``) (via mirror)
     - `McAuley Lab <https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data>`_
     - **No licence stated** (citation requested)
     - Not granted
     - **Not granted**
     - automatic
     - Kang & McAuley, ICDM 2018; Wan & McAuley, RecSys 2018; Pathak et al., SIGIR 2017
   * - ⚠ Netflix Prize (``netflix``)
     - Internet Archive: an **unofficial third-party upload** of the original release
     - Original Netflix Prize README: research only; no endorsement; acknowledge in publications **and send Netflix a copy**
     - No, without permission
     - **No**, without separate permission
     - automatic, from the unofficial mirror
     - Bennett & Lanning, KDD Cup 2007
   * - ⚠ MSD Taste Profile (``taste-profile``)
     - `Million Song Dataset <http://millionsongdataset.com/tasteprofile/>`_
     - Echo Nest API terms of use (page **no longer available**). MSD FAQ: researchers publishing results are fine; companies should ask.
     - Ask (effectively no)
     - Not addressed
     - automatic, **plain HTTP**
     - Bertin-Mahieux et al., ISMIR 2011
   * - Gowalla (``gowalla``)
     - `SNAP <https://snap.stanford.edu/data/loc-gowalla.html>`_
     - No licence stated (citation requested)
     - Not stated
     - Not stated
     - automatic
     - Cho, Myers & Leskovec, KDD 2011; SNAP collection (Leskovec & Krevl 2014)
   * - Retailrocket (``retailrocket``)
     - `Kaggle <https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset>`_
     - CC BY-NC-SA 4.0
     - No
     - Non-commercial, attribution, share-alike
     - **manual** (Kaggle account)
     - attribution to Retailrocket
   * - Music4All-Onion (``music4all-onion``)
     - `Zenodo 6609677 <https://zenodo.org/records/6609677>`_
     - CC BY 4.0
     - Yes
     - Yes, with attribution
     - automatic
     - Moscati et al., CIKM 2022
   * - OTTO (``otto``)
     - `Kaggle <https://www.kaggle.com/datasets/otto/recsys-dataset>`_ / `GitHub <https://github.com/otto-de/recsys-dataset>`_
     - CC BY 4.0 (data); MIT (code)
     - Yes
     - Yes, with attribution
     - **manual** (Kaggle CLI)
     - BibTeX in the OTTO repository
   * - Yambda (``yambda``)
     - `Hugging Face <https://huggingface.co/datasets/yandex/yambda>`_
     - Apache 2.0
     - Yes
     - Yes (with licence notice)
     - automatic
     - per the dataset card
   * - DBbook (``dbbook``)
     - `SWAP release, Zenodo 15403972 <https://zenodo.org/records/15403972>`_
     - CC BY 4.0 (Zenodo record, covering interactions and features as redistributed by SWAP); the original ESWC 2014 challenge release carried no licence
     - Yes, under CC BY 4.0 (see note)
     - Yes, under CC BY 4.0 (see note)
     - automatic
     - Spillo et al. (Zenodo 15403972); ESWC 2014 LOD-RecSys challenge for origin
   * - ⚠ Last.fm 2K (``lfm2k``) (via mirror)
     - `HetRec 2011 <https://grouplens.org/datasets/hetrec-2011/>`_, downloaded from the SWAP copy (Zenodo 15403972)
     - HetRec README: **non-commercial use only**. The SWAP copy is labelled CC BY 4.0, which cannot lift Last.fm's condition.
     - No (contact Last.fm)
     - Not addressed at origin; treat as non-commercial
     - automatic
     - Last.fm website; Cantador, Brusilovsky & Kuflik, HetRec'11

**Notes:**

- **Netflix Prize.** Netflix withdrew the dataset in 2010 after re-identification
  research and a privacy lawsuit. The library downloads it from an Internet Archive
  copy uploaded by a third party, which Netflix did not authorise. **Use it at your
  own responsibility.** We recommend it only for comparison with published results,
  and never publishing Netflix-derived checkpoints or per-user artefacts.
- **DBbook provenance.** SWAP redistributes data it did not originally create.
  CC BY 4.0 is the best available grant, but carries that residual uncertainty.
- **Text features** for ML-1M, ML-20M and Goodbooks come from the
  `beeFormer repository <https://github.com/recombee/beeformer>`_ (CC BY-SA 4.0).
  The item descriptions were generated with Llama 3.1, so Meta's Llama 3.1 licence
  also applies to them ("Built with Llama").
- **Gowalla** contains per-user location check-ins (personal data in origin).

Pretrained multimodal features
------------------------------

The SWAP feature archives for ML-1M, DBbook and Last.fm 2K are downloaded from
`Zenodo 15403972 <https://zenodo.org/records/15403972>`_ under **CC BY 4.0** and
are MD5-verified. The underlying media (posters, covers, audio) are not
downloaded. That record also redistributes ML-1M interaction files, which the
GroupLens terms do not permit without permission. Compresso Recsys therefore
always takes ML-1M interactions from GroupLens, not from the Zenodo copy. Cite the
encoders as listed in :doc:`datasets`.

Integrity of downloads
----------------------

Only the SWAP feature archives are verified against a published checksum.
Interaction downloads are checked for size, not content, but every checkpoint
built since 0.4 records each raw source file's size and SHA-256 in its manifest
(``cr.load_provenance``), so a build can be compared with a published one.
Taste Profile is fetched over plain HTTP, and MovieLens uses a temporary fallback
for GroupLens's expired TLS certificate (:doc:`datasets`).
