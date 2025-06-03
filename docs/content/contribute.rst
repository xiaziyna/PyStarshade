Contributing
============

We welcome any help—bug fixes, feature proposals, usage questions (xiaziyna at gmail dot com).

Report Bugs & Problems
----------------------

1. Check for duplicates: `existing issues <https://github.com/xiaziyna/PyStarshade/issues>`_.  
2. If none match, open a new issue and include:
   - **PyStarshade version**: ``pip show pystarshade``  
   - **Python version & OS**  
   - **Repro steps** & minimal code snippet  
   - **Error messages** or tracebacks  

.. note::
   Tag it with **bug** if appropriate.

Request Features
----------------

Open a new issue, choose the **question** label, and describe what you want + why. 

Contribute Code
---------------

#. **Fork** the repo and branch from ``main``::

      git clone git@github.com:xiaziyna/PyStarshade.git
      cd PyStarshade
      git checkout -b feature/short-name

#. Follow style:
   - Add or update tests in ``tests/``  
   - Verify all tests pass with ``pytest``

#. Commit with clear messages, e.g.::

      feat: add widget-alignment option
#. Push your branch and open a **Pull Request** against ``main``:
   - Explain your changes and why  

Guidelines
-----------

- Keep requests limited in scope and explain why your change enhances the utility of the tool.

