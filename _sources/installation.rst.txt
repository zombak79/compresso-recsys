Installation
============

Compresso Recsys is published as the ``compresso-recsys`` distribution and
imported as ``compresso_recsys`` in Python code.

Local Development
-----------------

When developing next to a checkout of Compresso, install both packages in
editable mode:

.. code-block:: bash

   pip install -e ../compresso
   pip install -e ".[test,sbert]"

Install from GitHub
-------------------

Once both repositories are public, install directly from GitHub:

.. code-block:: bash

   pip install "compresso-recsys@git+https://github.com/zombak79/compresso-recsys.git"

Optional Extras
---------------

Dataset and SBERT integrations are optional:

.. code-block:: bash

   pip install -e ".[datasets]"
   pip install -e ".[sbert]"

Build the Documentation Locally
-------------------------------

From the ``compresso-recsys`` project directory:

.. code-block:: bash

   pip install -r docs/requirements.txt
   pip install -e .
   sphinx-build -b html docs/source docs/build/html

The generated HTML will be available in ``docs/build/html``.
