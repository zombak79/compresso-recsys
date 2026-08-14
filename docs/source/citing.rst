Citing Compresso Recsys
=======================

If you use the recommender models in academic work, please consider citing the
papers that introduced the corresponding methods.

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
