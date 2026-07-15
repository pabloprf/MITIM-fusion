PORTALS
=======

The PORTALS method, described in `P. Rodriguez-Fernandez et al.,  Nucl. Fusion 64 076034 (2024). <https://iopscience.iop.org/article/10.1088/1741-4326/ad4b3d>`_ consists of using Bayesian Optimization techniques to find steady-state solutions of transport codes of arbitrary fidelity.

Once setup has been successful, the following regression test should run smoothly:

.. code-block:: console

   python3 MITIM-fusion/tests/capability_tests/portals_01_tglf_standard.py

.. warning::

   It is recommended that you run first the regression tests for the general optimization workflow and the TGLF interface:
   
   .. code-block:: console

      python3 MITIM-fusion/tests/capability_tests/optimization_01_custom_function.py

      python3 MITIM-fusion/tests/capability_tests/tglf_02_run_from_tglfinput.py

   If both of those tests work, it is highly likely that the PORTALS workflow will work as well.


Run a standard profile prediction with PORTALS-TGLF
---------------------------------------------------

*Under Development*

*(In the meantime, please checkout* `tests/capability_tests/portals_01_tglf_standard.py <https://github.com/pabloprf/MITIM-fusion/blob/main/tests/capability_tests/portals_01_tglf_standard.py>`_ *)*

Run a standard profile prediction with PORTALS-CGYRO
----------------------------------------------------

*Under Development*
