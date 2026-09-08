Citing Compresso Recsys
=======================

If you use Compresso Recsys in academic work, cite the methods, datasets, and
additional metadata sources used in your experiment. Dataset citations and
preprocessing-method citations serve different purposes; include both when
applicable. See :doc:`datasets` for the actual sources and loader defaults.

.. contents:: On this page
   :local:
   :depth: 2

Compresso
---------

If your work uses the Compresso sparse-representation framework, cite the
project:

.. code-block:: bibtex

   @misc{compresso,
     title  = {Compresso: A PyTorch Framework for Sparse Representation Learning},
     author = {Van{\v{c}}ura, Vojt{\v{e}}ch and Giacomo Medda and Spi{\v{s}}{\'a}k, Martin and Ladislav Pe{\v{s}}ka},
     year   = {2026},
     url    = {https://github.com/zombak79/compresso}
   }

EASE
----

For :class:`compresso_recsys.models.EASE`, cite the original EASE paper:

.. code-block:: bibtex

   @inproceedings{steck2019embarrassingly,
     title={Embarrassingly shallow autoencoders for sparse data},
     author={Steck, Harald},
     booktitle={The World Wide Web Conference},
     pages={3251--3257},
     year={2019}
   }

Neighborhood Models
-------------------

For :class:`compresso_recsys.models.UserKNNRecommender`, cite the GroupLens
user-based collaborative-filtering paper:

.. code-block:: bibtex

   @inproceedings{resnick1994grouplens,
     title={GroupLens: An Open Architecture for Collaborative Filtering of Netnews},
     author={Resnick, Paul and Iacovou, Neophytos and Suchak, Mitesh and
             Bergstrom, Peter and Riedl, John},
     booktitle={Proceedings of the 1994 ACM Conference on Computer Supported
                Cooperative Work},
     pages={175--186},
     year={1994},
     doi={10.1145/192844.192905}
   }

For :class:`compresso_recsys.models.ItemKNNRecommender`, cite the original
item-based collaborative-filtering paper:

.. code-block:: bibtex

   @inproceedings{sarwar2001item,
     title={Item-Based Collaborative Filtering Recommendation Algorithms},
     author={Sarwar, Badrul and Karypis, George and Konstan, Joseph and
             Riedl, John},
     booktitle={Proceedings of the 10th International Conference on World Wide Web},
     pages={285--295},
     year={2001},
     doi={10.1145/371920.372071}
   }

Mult-VAE, Mult-DAE, and AutoRec
-------------------------------

For :class:`compresso_recsys.models.MultDAETrainer` or
:class:`compresso_recsys.models.MultVAETrainer`, cite the paper that introduced
both models:

.. code-block:: bibtex

   @inproceedings{liang2018variational,
     title={Variational Autoencoders for Collaborative Filtering},
     author={Liang, Dawen and Krishnan, Rahul G. and Hoffman, Matthew D. and
             Jebara, Tony},
     booktitle={Proceedings of the 2018 World Wide Web Conference},
     pages={689--698},
     year={2018},
     doi={10.1145/3178876.3186150}
   }

AutoRec established the collaborative-filtering autoencoder architecture that
preceded Mult-DAE:

.. code-block:: bibtex

   @inproceedings{sedhain2015autorec,
     title={AutoRec: Autoencoders Meet Collaborative Filtering},
     author={Sedhain, Suvash and Menon, Aditya Krishna and Sanner, Scott and
             Xie, Lexing},
     booktitle={Proceedings of the 24th International Conference on World Wide Web},
     pages={111--112},
     year={2015},
     doi={10.1145/2740908.2742726}
   }

TEASER
------

For :class:`compresso_recsys.models.TEASER` or
:class:`compresso_recsys.models.TEASERGDTrainer`, cite the original TEASER
paper:

.. code-block:: bibtex

   @inproceedings{depauw2022who,
     title={Who do you think I am? Interactive User Modelling with Item Metadata},
     author={De Pauw, Joey and Ruymbeek, Koen and Goethals, Bart},
     booktitle={Proceedings of the 16th ACM Conference on Recommender Systems},
     pages={640--643},
     year={2022},
     doi={10.1145/3523227.3551470}
   }

ELSA
----

For standard :class:`compresso_recsys.models.ELSA` training, cite the original
ELSA paper:

.. code-block:: bibtex

   @inproceedings{vanvcura2022scalable,
     title={Scalable linear shallow autoencoder for collaborative filtering},
     author={Van{\v{c}}ura, Vojt{\v{e}}ch and Alves, Rodrigo and Kasalick{\`y}, Petr and Kord{\'\i}k, Pavel},
     booktitle={Proceedings of the 16th ACM conference on recommender systems},
     pages={604--609},
     year={2022}
   }

ELSA with sampled output candidates
-----------------------------------

If your work uses ``ELSAConfig.max_output`` to train against sampled output
candidates at scale, also consider citing the large-scale ELSA evaluation:

.. code-block:: bibtex

   @article{10.1145/3748335,
     author = {Van\v{c}ura, Vojt\v{e}ch and Kasalick\'{y}, Petr and Alves, Rodrigo and Kord\'{\i}k, Pavel},
     title = {Evaluating Linear Shallow Autoencoders on Large Scale Datasets},
     year = {2025},
     publisher = {Association for Computing Machinery},
     address = {New York, NY, USA},
     url = {https://doi.org/10.1145/3748335},
     doi = {10.1145/3748335},
     journal = {ACM Trans. Recomm. Syst.},
   }

Compressed ELSA
---------------

If your work uses :class:`compresso_recsys.models.CompressedELSA`, cite the
sparse-representation paper. The original ELSA citation above may also be
appropriate when ELSA itself is central to the work.

.. code-block:: bibtex

   @inproceedings{vanvcura2026efficient,
     title={Efficient Learning of Sparse Representations from Interactions},
     author={Van{\v{c}}ura, Vojt{\v{e}}ch and Spi{\v{s}}{\'a}k, Martin and Alves, Rodrigo and Pe{\v{s}}ka, Ladislav},
     booktitle={Proceedings of the ACM Web Conference 2026},
     pages={8577--8580},
     year={2026}
   }

.. _cite-sasrec:

SASRec
------

For :class:`compresso_recsys.models.SASRecTrainer`, cite the original SASRec
paper. Its sequential objective, tied item scoring, and published MovieLens
hyperparameters are SASRec-derived. The transformer block uses a modernized
pre-norm attention architecture, so published results are a point of comparison
rather than exact implementation parity:

.. code-block:: bibtex

   @inproceedings{kang2018self,
     title={Self-Attentive Sequential Recommendation},
     author={Kang, Wang-Cheng and McAuley, Julian},
     booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
     pages={197--206},
     year={2018},
     doi={10.1109/ICDM.2018.00035}
   }

SimpleRNN
---------

For :class:`compresso_recsys.models.SimpleRNNTrainer`, GRU4Rec is the relevant
recurrent recommendation reference. Compresso's model supports GRU or LSTM
encoders and uses full-catalog next-item cross-entropy on user histories;
it does not reproduce GRU4Rec's session-parallel batching or original ranking
losses. Cite this paper as methodological background and describe the actual
configuration used:

.. code-block:: bibtex

   @inproceedings{hidasi2016session,
     title={Session-based Recommendations with Recurrent Neural Networks},
     author={Hidasi, Balazs and Karatzoglou, Alexandros and Baltrunas, Linas and Tikk, Domonkos},
     booktitle={International Conference on Learning Representations},
     year={2016},
     url={https://arxiv.org/abs/1511.06939}
   }

SimpleGPT and SimpleBidirectionalTransformer
----------------------------------------------

:class:`compresso_recsys.models.SimpleGPTTrainer` uses a causal Transformer
with a learned prefix and tied item embeddings by default. Its implementation
follows the nanoGPT architecture, adapted to item sequences.
:class:`compresso_recsys.models.SimpleBidirectionalTransformerTrainer` reuses
those building blocks with bidirectional attention and a pooled ``CLS`` state
to predict an unordered target set. Cite the Transformer foundation, and
attribute nanoGPT when discussing the implementation:

.. code-block:: bibtex

   @inproceedings{vaswani2017attention,
     title={Attention Is All You Need},
     author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob
             and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
     booktitle={Advances in Neural Information Processing Systems},
     volume={30},
     year={2017},
     url={https://arxiv.org/abs/1706.03762}
   }

   @misc{karpathy_nanogpt,
     author={Karpathy, Andrej},
     title={{nanoGPT}},
     howpublished={GitHub repository},
     url={https://github.com/karpathy/nanoGPT},
     note={Record the revision used in your experiment}
   }

These are Compresso baseline implementations, not separate published methods.
In particular, the bidirectional model is not an implementation of BERT4Rec's
masked-item objective. Its set-valued multinomial loss is related to the
Mult-VAE/Mult-DAE formulation above; cite that work when the loss is relevant.

Content, Popularity, and Random Baselines
-------------------------------------------

:class:`compresso_recsys.models.ContentRecommender` uses item features and
similarity scoring; cite the feature/embedding model actually used to create
those features. :class:`compresso_recsys.models.PopularityBaseline` and
:class:`compresso_recsys.models.RandomBaseline` are elementary baselines with
no single originating paper attributed by this package. Report their settings
and the Compresso Recsys version or commit. Catalog adapters and model base
classes are infrastructure, not additional recommendation methods.

.. _cite-datasets:

Datasets
--------

MovieLens 1M and 20M
~~~~~~~~~~~~~~~~~~~~~~

For :class:`compresso_recsys.datasets.MovieLens1M` and
:class:`compresso_recsys.datasets.MovieLens20M`, cite the dataset history paper
and state the exact variant. GroupLens usage terms remain applicable.

.. code-block:: bibtex

   @article{harper2015movielens,
     title={The MovieLens Datasets: History and Context},
     author={Harper, F. Maxwell and Konstan, Joseph A.},
     journal={ACM Transactions on Interactive Intelligent Systems},
     volume={5},
     number={4},
     year={2015},
     doi={10.1145/2827872},
     url={https://doi.org/10.1145/2827872}
   }

Goodbooks-10k
~~~~~~~~~~~~~

For :class:`compresso_recsys.datasets.Goodbooks`, credit the
`Goodbooks-10k release <https://github.com/zygmuntz/goodbooks-10k/releases/tag/v1.0>`_.
The upstream repository supplies the dataset rather than a designated research
paper; use a dataset/software citation and retain its attribution and usage
terms. Goodbooks-10k is distinct from McAuley Lab's Goodreads datasets.

.. code-block:: bibtex

   @misc{zygmuntz2017goodbooks,
     author={{zygmuntz}},
     title={{Goodbooks-10k}: Ten Thousand Books, Six Million Ratings},
     year={2017},
     howpublished={Dataset, release v1.0},
     url={https://github.com/zygmuntz/goodbooks-10k}
   }

Amazon Reviews 2023
~~~~~~~~~~~~~~~~~~~~~

For :class:`compresso_recsys.datasets.AmazonReviews2023`, use the citation
recommended by the `official dataset page <https://amazon-reviews-2023.github.io/>`_.
Report the category, source revision, and filtering; citations for older Amazon
releases do not identify this collection.

.. code-block:: bibtex

   @article{hou2024bridging,
     title={Bridging Language and Items for Retrieval and Recommendation},
     author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and
             Chen, Xiusi and McAuley, Julian},
     journal={arXiv preprint arXiv:2403.03952},
     year={2024},
     url={https://arxiv.org/abs/2403.03952}
   }

Steam
~~~~~

For :class:`compresso_recsys.datasets.Steam`, the
`upstream Steam page <https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data>`_
lists SASRec (:ref:`cite-sasrec`) and the two references below. Report that the
adapter uses ``steam_reviews.json.gz`` and ``steam_games.json.gz``: it does
not load bundle interactions or the BERT4Rec remapped sequence release.

.. code-block:: bibtex

   @inproceedings{wan2018monotonic,
     title={Item Recommendation on Monotonic Behavior Chains},
     author={Wan, Mengting and McAuley, Julian},
     booktitle={Proceedings of the 12th ACM Conference on Recommender Systems},
     year={2018},
     url={https://cseweb.ucsd.edu/~jmcauley/pdfs/recsys18b.pdf}
   }

   @inproceedings{pathak2017steam,
     title={Generating and Personalizing Bundle Recommendations on Steam},
     author={Pathak, Apurva and Gupta, Kshitiz and McAuley, Julian},
     booktitle={Proceedings of the 40th International ACM SIGIR Conference on
                Research and Development in Information Retrieval},
     year={2017},
     url={https://cseweb.ucsd.edu/~jmcauley/pdfs/sigir17.pdf}
   }

Netflix Prize
~~~~~~~~~~~~~

For :class:`compresso_recsys.datasets.NetflixPrize`, cite the original dataset
description. Internet Archive is the adapter's download location, not the
dataset's creator. Original Netflix terms remain applicable.

.. code-block:: bibtex

   @inproceedings{bennett2007netflix,
     title={The Netflix Prize},
     author={Bennett, James and Lanning, Stan},
     booktitle={Proceedings of KDD Cup and Workshop 2007},
     year={2007},
     url={https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/The-Netflix-Prize-Bennett.pdf}
   }

MSD / Taste Profile
~~~~~~~~~~~~~~~~~~~~~

For :class:`compresso_recsys.datasets.TasteProfile`, cite the Million Song
Dataset paper and explicitly identify the **Taste Profile subset contributed
by The Echo Nest**, using ``train_triplets.txt``. The adapter loads user-song
play counts, not the full MSD audio features or song metadata.
See the `MSD source <http://millionsongdataset.com/>`_ for attribution.

.. code-block:: bibtex

   @inproceedings{bertinmahieux2011million,
     title={The Million Song Dataset},
     author={Bertin-Mahieux, Thierry and Ellis, Daniel P. W. and Whitman, Brian
             and Lamere, Paul},
     booktitle={Proceedings of the 12th International Society for Music
                Information Retrieval Conference},
     year={2011},
     url={https://ismir2011.ismir.net/papers/OS6-1.pdf}
   }

Gowalla
~~~~~~~

For :class:`compresso_recsys.datasets.Gowalla`, cite the paper requested by
`SNAP <https://snap.stanford.edu/data/loc-gowalla.html>`_. State that the adapter
uses raw check-ins; the social graph is not downloaded.

.. code-block:: bibtex

   @inproceedings{cho2011friendship,
     title={Friendship and Mobility: User Movement in Location-Based Social Networks},
     author={Cho, Eunjo and Myers, Seth A. and Leskovec, Jure},
     booktitle={Proceedings of the 17th ACM SIGKDD International Conference on
                Knowledge Discovery and Data Mining},
     year={2011},
     url={https://snap.stanford.edu/data/loc-gowalla.html}
   }

Additional Movie and Book Descriptions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The MovieLens and Goodbooks adapters also load generated descriptions from
beeFormer. If those descriptions contribute to your experiment's item features,
cite beeFormer in addition to the original dataset:

.. code-block:: bibtex

   @inproceedings{vancura2024beeformer,
     title={beeFormer: Bridging the Gap Between Semantic and Interaction Similarity
            in Recommender Systems},
     author={Van{\v{c}}ura, Vojt{\v{e}}ch and Kord{\'\i}k, Pavel and Straka, Milan},
     booktitle={Proceedings of the 18th ACM Conference on Recommender Systems},
     pages={1102--1107},
     year={2024},
     doi={10.1145/3640457.3691707},
     url={https://github.com/recombee/beeformer}
   }

Evaluation Protocol
-------------------

Held-out users are evaluated under strong generalization: each user's history is
split into a fold-in part the model sees and a held-out part it is scored
against. ``eval_holdout_frac`` defaults to 0.2, matching the 80/20 split
described by Liang et al. If you report numbers from ``user_split``, cite the
Mult-VAE/Mult-DAE paper above.

Stacking several independent draws per user — ``eval_draws``, defaulting to 5 —
follows the ELSA line of work; cite the ELSA papers above when reporting under
that protocol.

Dataset preprocessing
~~~~~~~~~~~~~~~~~~~~~

Netflix and Taste Profile support/feedback defaults follow the Mult-VAE paper
above. Steam's every-review-as-feedback convention follows BERT4Rec, while
Gowalla's 10-interaction CF support defaults follow NGCF:

.. code-block:: bibtex

   @inproceedings{sun2019bert4rec,
     title={{BERT4Rec}: Sequential Recommendation with Bidirectional Encoder
            Representations from Transformer},
     author={Sun, Fei and Liu, Jun and Wu, Jian and Pei, Changhua and Lin, Xiao
             and Ou, Wenwu and Jiang, Peng},
     booktitle={Proceedings of the 28th ACM International Conference on
                Information and Knowledge Management},
     year={2019},
     doi={10.1145/3357384.3357895},
     url={https://arxiv.org/abs/1904.06690}
   }

   @inproceedings{wang2019ngcf,
     title={Neural Graph Collaborative Filtering},
     author={Wang, Xiang and He, Xiangnan and Wang, Meng and Feng, Fuli and Chua, Tat-Seng},
     booktitle={Proceedings of the 42nd International ACM SIGIR Conference on
                Research and Development in Information Retrieval},
     year={2019},
     url={https://arxiv.org/abs/1905.08108}
   }

These citations explain the defaults, not exact reproduction of published
results. Report the actual ``split_mode``, support thresholds, seed, and metric
definitions. Compresso's LLO uses three stage targets; temporal splits use
global time windows; neither is the original BERT4Rec sampled-ranking protocol.
The raw Gowalla adapter does not use NGCF/LightGCN's published train/test files.

Statistical Comparison
----------------------

If you report significance from :mod:`compresso_recsys.stats`, cite the methods
it implements. See :doc:`statistical-comparison` for what each one contributes
and a ready-to-adapt methods paragraph.

The paired bootstrap behind the confidence interval:

.. code-block:: bibtex

   @article{efron1979bootstrap,
     title={Bootstrap Methods: Another Look at the Jackknife},
     author={Efron, Bradley},
     journal={The Annals of Statistics},
     volume={7},
     number={1},
     pages={1--26},
     year={1979},
     doi={10.1214/aos/1176344552}
   }

The randomization test used for the p-value, and the comparison of significance
tests that motivates choosing it for retrieval evaluation:

.. code-block:: bibtex

   @inproceedings{smucker2007comparison,
     title={A Comparison of Statistical Significance Tests for Information
            Retrieval Evaluation},
     author={Smucker, Mark D. and Allan, James and Carterette, Ben},
     booktitle={Proceedings of the Sixteenth ACM Conference on Information and
                Knowledge Management},
     series={CIKM '07},
     pages={623--632},
     year={2007},
     publisher={ACM},
     doi={10.1145/1321440.1321528}
   }

Why a Monte Carlo p-value is computed as ``(1 + extreme) / (B + 1)`` rather
than as a plain proportion:

.. code-block:: bibtex

   @article{phipson2010permutation,
     title={Permutation P-values Should Never Be Zero: Calculating Exact
            P-values When Permutations Are Randomly Drawn},
     author={Phipson, Belinda and Smyth, Gordon K.},
     journal={Statistical Applications in Genetics and Molecular Biology},
     volume={9},
     number={1},
     pages={Article 39},
     year={2010},
     doi={10.2202/1544-6115.1585}
   }

The multiple-testing correction applied across a comparison report:

.. code-block:: bibtex

   @article{holm1979simple,
     title={A Simple Sequentially Rejective Multiple Test Procedure},
     author={Holm, Sture},
     journal={Scandinavian Journal of Statistics},
     volume={6},
     number={2},
     pages={65--70},
     year={1979}
   }

The Monte Carlo p-value form and the convention of choosing the resample count
so that ``alpha * (B + 1)`` is an integer:

.. code-block:: bibtex

   @book{davison1997bootstrap,
     title={Bootstrap Methods and their Application},
     author={Davison, A. C. and Hinkley, D. V.},
     publisher={Cambridge University Press},
     year={1997},
     doi={10.1017/CBO9780511802843}
   }
.. _cite-swap-multimodal:

SWAP multimodal ML-1M, DBbook, and Last.fm-2K
----------------------------------------------------

For pretrained features, cite the resource paper and the versioned dataset.
Continue citing MovieLens for ML-1M. For DBbook, this adapter uses the reconstructed
interaction release described by SWAP, not a claim of retrieving the original
ESWC challenge service.

Paper: `See the Movie, Hear the Song, Read the Book
<https://doi.org/10.1145/3705328.3748162>`_.
Data: `Zenodo record 15403972 <https://zenodo.org/records/15403972>`_.
Processing: `authors' repository
<https://github.com/swapUniba/multimodal_ml1m_dbbook_lfm2k>`_.

.. code-block:: bibtex

   @inproceedings{spillo2025multimodal,
     author = {Spillo, Giuseppe and Musacchio, Elio and Musto, Cataldo and
               de Gemmis, Marco and Lops, Pasquale and Semeraro, Giovanni},
     title = {See the Movie, Hear the Song, Read the Book: Extending
              MovieLens-1M, Last.fm-2K, and DBbook with Multimodal Data},
     booktitle = {Proceedings of the Nineteenth ACM Conference on Recommender Systems},
     year = {2025},
     doi = {10.1145/3705328.3748162}
   }

   @misc{spillo2025multimodaldata,
     author = {Spillo, Giuseppe and Musacchio, Elio and Musto, Cataldo and
               de Gemmis, Marco and Lops, Pasquale and Semeraro, Giovanni},
     title = {See the Movie, Hear the Song, Read the Book: Extending
              MovieLens-1M, Last.fm 2K, and DBBook with multimodal Data},
     year = {2025},
     publisher = {Zenodo},
     doi = {10.5281/zenodo.15403972}
   }

For Last.fm, the original README requests attribution to `Last.fm
<https://www.last.fm/>`_ and suggests the HetRec workshop citation below.
See `GroupLens HetRec 2011 <https://grouplens.org/datasets/hetrec-2011/>`_
and the supplied README for usage terms; availability in a public archive does
not waive the original dataset restrictions.

.. code-block:: bibtex

   @inproceedings{cantador2011hetrec,
     author = {Cantador, Iv{\'a}n and Brusilovsky, Peter and Kuflik, Tsvi},
     title = {2nd Workshop on Information Heterogeneity and Fusion in
              Recommender Systems (HetRec 2011)},
     booktitle = {Proceedings of the 5th ACM Conference on Recommender Systems},
     year = {2011},
     publisher = {ACM}
   }
