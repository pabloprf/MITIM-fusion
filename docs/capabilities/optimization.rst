Optimization Capabilities
=========================

**MITIM** can be used to optimize any custom function (:ref:`Optimize a custom function`) or simulations that have already been developed in the code (:ref:`Fusion applications`), such as :ref:`VITALS` and :ref:`PORTALS`.
Make sure you follow the :ref:`Installation` tutorial for information on how to get MITIM working and how to configure your setup.

Once setup has been successful, the following regression test should run smoothly:

.. code-block:: console

   python3 MITIM-fusion/tests/OPT_workflow.py

.. contents:: Contents
    :local:
    :depth: 3

Fusion applications
-------------------

.. toctree:: 
   :maxdepth: 1

   vitals_capabilities
   portals_capabilities
   freegsu_capabilities


Optimize a custom function
--------------------------

Optimizing any function (mathematical or a simulation) with MITIM is very easy.

For this tutorial we will need the following modules:

.. code-block:: python

   import torch
   import numpy as np
   from pathlib import Path
   from mitim_tools.opt_tools     import STRATEGYtools

Select the location of the MITIM namelist (see :ref:`Understanding the MITIM namelist` to understand how to construct the namelist file) and the folder to work on:

.. code-block:: python

   folder    = Path('MITIM-fusion/tests/scratch/mitim_tut')
   namelist  = Path('MITIM-fusion/templates/main.namelist.json')

Then create your custom optimization object as a child of the parent ``STRATEGYtools.opt_evaluator`` class.
You only need to modify what operations need to occur inside the ``run()`` (where operations/simulations happen) and ``scalarized_objective()`` (to define what is the target to maximize) methods.
In this example, we are using ``x**2`` as our function with a 2% evaluation error, to find ``x`` such that ``x**2=15``:

.. code-block:: python

   class opt_class(STRATEGYtools.opt_evaluator):
      def __init__(self, folder, namelist):
         # Store folder, namelist. Read namelist
         super().__init__(folder, namelist=namelist)
         # ----------------------------------------

         # Problem description (rest of problem parameters are taken from namelist)
         self.optimization_options["problem_options"]["dvs"] = ["x"]
         self.optimization_options["problem_options"]["dvs_min"] = [0.0]
         self.optimization_options["problem_options"]["dvs_max"] = [20.0]

         self.optimization_options["problem_options"]["ofs"] = ["z", "zval"]
         self.name_objectives = ["zval_match"]

      def run(self, paramsfile, resultsfile):
         # Read stuff
         folderEvaluation, numEval, dictDVs, dictOFs = self.read(paramsfile, resultsfile)

         # Operations
         dictOFs["z"]["value"] = dictDVs["x"]["value"] ** 2
         dictOFs["z"]["error"] = dictOFs["z"]["value"] * 2e-2

         dictOFs["zval"]["value"] = 15.0
         dictOFs["zval"]["error"] = 0.0

         # Write stuff
         self.write(dictOFs, resultsfile)

      def scalarized_objective(self, Y):
         ofs_ordered_names = np.array(self.optimization_options["problem_options"]["ofs"])

         of = Y[..., ofs_ordered_names == "z"]
         cal = Y[..., ofs_ordered_names == "zval"]

         # Residual is defined as the negative (bc it's maximization) normalized (1/N) norm of radial & channel residuals -> L1
         res = -1 / of.shape[-1] * torch.norm((of - cal), p=1, dim=-1)

         return of, cal, res

Then, create an object from the previously defined class:

.. code-block:: python

   opt_fun1D  = opt_class(folder,namelist=namelist)

.. tip::

   Note that at this point, you can pass any parameter that you want, just changing the ``__init__()`` method as appropriate.

Now we can create and launch the MITIM optimization process from the beginning (i.e. ``cold_start = True``):

.. code-block:: python

   MITIM_BO = STRATEGYtools.MITIM_BO( opt_fun1D, cold_startYN = True )
   MITIM_BO.run()

Once finished, we can plot the results easily with:

.. code-block:: python

   opt_fun1D.plot_optimization_results(analysis_level=2)


Understanding the MITIM namelist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checkout file ``MITIM-fusion/templates/main.namelist.json``, which has comprehensive comments.

*Under development*

Understanding the MITIM outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As a result of the last step of :ref:`Optimize a custom function`, optimization results are plotted...

*Under development*
