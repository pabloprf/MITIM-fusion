============
Installation
============

.. contents::
	:local:
	:depth: 1

.. attention::
   MITIM requires python>=3.9, a requirement driven by the optimization capabilities in PyTorch.
   If you do not have python>=3.9 but still want to use MITIM's non-optimization features, you may try to install each python package individually (see ``setup.py`` file) and skip ``botorch``. However, this option is not supported and there is no assurance the code will work.

Instructions
------------

Clone the `GitHub repository  <https://github.com/pabloprf/MITIM-fusion>`_ (do not forget to select **Watch All activity** to receive notifications of new releases and announcements, and **Star** the repository to increase its visibility):

.. code-block:: console

   git clone git@github.com:pabloprf/MITIM-fusion.git

Source the configuration file (it may be good to add this in your ``.bashrc`` file to load upon login, or with some alias to initialize ``mitim``) by 
pointing to the folder where you cloned the repository:

.. code-block:: console

   export MITIM_PATH=/path/to/mitim/
   source $MITIM_PATH/config/mitim.bashrc
   
.. hint::
   
   It may be useful, at this point, to create a virtual environment to install required MITIM dependencies. For example, using python's ``venv`` package:

   .. code-block:: console

      python3.9 -m venv mitim-env
      source mitim-env/bin/activate
      pip3 install pip --upgrade

Use ``pip3`` to install all the required MITIM requirements:

.. code-block:: console

   pip3 install -e $MITIM_PATH[pyqt]

.. note::
   
   The optional argument ``[pyqt]`` added in the intallation command above must only be used if the machine allows for graphic interfaces.
   If running in a computing cluster, remove that flag.
   The ``pyqt`` package is used to create condensed figures into a single notebook when interpreting and plotting simulation results.
   
   If you wish to install all capabilities (including compatibility with the `OMFIT <https://omfit.io/>`_  or `FREEGS <https://github.com/freegs-plasma/freegs>`_ codes), it is recommended that ``pip3`` is run as follows:

   .. code-block:: console

      pip3 install -e $MITIM_PATH[pyqt,omfit,freegs]


If you were unsuccessful in the installation, check out our :ref:`Frequently Asked Questions` section.


User configuration
------------------

In ``$MITIM_PATH/config/``, there is a ``config_user_example.json`` with specifications of where to run certain codes and what the login requirements are. **If you are planning on using MITIM to run plasma simulation codes**, please create an equivalent file ``config_user.json`` in the same folder, indicating your specific needs.

.. code-block:: console

   cp $MITIM_PATH/config/config_user_example.json $MITIM_PATH/config/config_user.json
   vim $MITIM_PATH/config/config_user.json

Apart from machine configurations, ``preferences`` in ``config_user.json`` also includes a ``verbose_level`` flag, which indicates the amount of messages that are printed to the terminal when running MITIM.
For debugging purposes, it is recommended a maximum verbose level of ``5``.
For production runs, a minimum verbose level of ``1`` is recommended so that you only get important messages.
``preferences`` also allows a ``dpi_notebook`` value (in percent from standard), which should be adjusted for each user's screen configuration.

For example, if TGLF is set up to run in the *eofe7.mit.edu* machine, this means that, every time in the MITIM workflow when TGLF needs to run, it will access *eofe7.mit.edu* machine to do so, and therefore you must specify how to access the engaging machine:

.. code-block:: console

      "preferences": {
         "tglf":             "engaging",
         "verbose_level":    "5",
         "dpi_notebook":     "80"
      },
      "engaging": {
         "machine":          "eofe7.mit.edu", 
         "username":         "pablorf",
         "partition":        "sched_mit_psfc",
         "identity":         "~/.ssh/id_rsa",
         "scratch":          "/nobackup1/pablorf/scratch/"
         }

If you select to run a code in a given machine, please make sure you have ssh rights to that machine with the login instructions specified, unless you are running it locally.
MITIM will attempt to secure-copy and access that machine through a standard SSH connection and it must therefore be set-up prior to launching MITIM jobs.
Make sure that you can ssh with ``ssh username@machine`` with no password for the SSH keys or via a proxy connection (otherwise MITIM will ask for the password very often).

.. attention::

   Note that MITIM does not maintain or develop the simulation codes that are used within it, such as those from `GACODE <http://gafusion.github.io/doc/index.html>`_ or `TRANSP <hhttps://transp.pppl.gov/index.html>`_. It assumes that proper permissions have been obtained and that working versions of those codes exist in the machine configured to run them.

License and contributions
-------------------------

MITIM is released under the MIT License, one of the most permissive and widely used open-source software licenses.
Our choice of this license aims to make the package as useful and applicable as possible, in support of the development of fusion energy.
Embracing the spirit of open-source collaboration, we appreciate users who help increase the visibility of our project by
starring the `MITIM-fusion <https://github.com/pabloprf/MITIM-fusion/>`_ GitHub repository and support and acknowledge the continuous development of this tool by citing the following works in any publications, talks and posters:

**[1]** P. Rodriguez-Fernandez, N.T. Howard, A. Saltzman, S. Kantamneni, J. Candy, C. Holland, M. Balandat, S. Ament and A.E. White, `Enhancing predictive capabilities in fusion burning plasmas through surrogate-based optimization in core transport solvers <https://arxiv.org/abs/2312.12610>`_, arXiv:2312.12610 (2023).

**[2]** P. Rodriguez-Fernandez, N.T. Howard and J. Candy, `Nonlinear gyrokinetic predictions of SPARC burning plasma profiles enabled by surrogate modeling <https://iopscience.iop.org/article/10.1088/1741-4326/ac64b2>`_, Nucl. Fusion 62, 076036 (2022).

These publications provide foundational insights and methodologies that have significantly contributed to the development of MITIM.

License
~~~~~~~

.. literalinclude:: LICENSE
   :language: text