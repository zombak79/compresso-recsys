Welcome to Compresso Recsys's documentation!
============================================

Compresso Recsys is a toolkit for practical recommender development, and the
recommender companion of Compresso. It has three parts you can use alone or
together:

1. **One data checkpoint:** the interaction matrix, sequences, item metadata,
   embeddings and splits in one portable file. It is built with a single
   command (:doc:`cli-reference`) and can carry results and model checkpoints.
2. **A model base class** that handles the boring parts around your model's
   own ``fit``: ``predict``, ``predict_on_batch`` batching, ``recommend`` with
   stable item IDs and filters, data handling, and save and load
   (:doc:`implementing-a-recommender`).
3. **Evaluation with statistical testing** as the final arbiter between models
   (:doc:`statistical-comparison`).

The project distribution is named ``compresso-recsys`` and the Python package
is imported as ``compresso_recsys``.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   getting-started
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   examples
   datasets
   licenses
   bring-your-own-dataset
   implementing-a-recommender
   adding-a-metric
   multimodal-concat-wrapper
   clcrec
   ccfcrec
   reproducing-sasrec-results-on-ml1m
   reproducing-bert4rec-results-on-ml1m
   
   statistical-comparison
   paper-reproduction
   citing
   cli-reference

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/index

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
