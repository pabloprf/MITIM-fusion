PORTALS
=======

The PORTALS method, described in `P. Rodriguez-Fernandez et al., Nulc. Fusion (2022) <https://iopscience.iop.org/article/10.1088/1741-4326/ac64b2>`_ consists of using Bayesian Optimization techniques to find steady-state solutions of transport codes of arbitrary fidelity.

Once setup has been successful, the following regression test should run smoothly:

.. code-block:: console

   python3 $MITIM_PATH/tests/PORTALS_workflow.py

.. warning::

   It is recommended that you run first the regression tests for the general optimization workflow and the TGLF interface:
   
   .. code-block:: console

      python3 $MITIM_PATH/tests/OPT_workflow.py

      python3 $MITIM_PATH/tests/TGLF_workflow.py

   If both of those tests work, it is highly likely that the PORTALS workflow will work as well.


Run a standard profile prediction with PORTALS-TGLF
---------------------------------------------------

*Under Development*

*(In the meantime, please checkout* `tutorials/PORTALS_tutorial.py <../../tutorials/PORTALS_tutorial.py>`_ *)*

Run a standard profile prediction with PORTALS-CGYRO
----------------------------------------------------

*Under Development*
