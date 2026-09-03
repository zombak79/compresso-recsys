Installation
============

Compresso Recsys is published as the ``compresso-recsys`` distribution and
imported as ``compresso_recsys`` in Python code.

Install from PyPI
-----------------

.. code-block:: bash

   pip install compresso-recsys

Optional Extras
---------------

Hugging Face dataset export is optional:

.. code-block:: bash

   pip install "compresso-recsys[datasets]"

The user-user and item-item KNN recommenders use scikit-learn for neighbor
search:

.. code-block:: bash

   pip install "compresso-recsys[knn]"

Local Development
-----------------

When developing next to a checkout of Compresso, install both packages in
editable mode:

.. code-block:: bash

   pip install -e ../compresso
   pip install -e ".[dev,datasets,test]"

Install from GitHub
-------------------

Install directly from GitHub when testing unreleased changes:

.. code-block:: bash

   pip install "compresso-recsys@git+https://github.com/zombak79/compresso-recsys.git"

Build the Documentation Locally
-------------------------------

From the ``compresso-recsys`` project directory:

.. code-block:: bash

   pip install -e ".[docs]"
   sphinx-build -b html docs/source docs/build/html

The generated HTML will be available in ``docs/build/html``.
