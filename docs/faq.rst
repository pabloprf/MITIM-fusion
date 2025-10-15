Frequently Asked Questions
==========================

.. contents::
   :local:
   :depth: 1

Issues during MITIM installation
--------------------------------

.. dropdown:: ``pyqt`` fails to install 

   Some ``pyqt`` installation problems can be fixed with upgrading the ``pip`` package manager:
   
   .. code-block:: console
      
      pip3 install pip --upgrade

.. dropdown:: Square brackets not understood by shell

   If you are using ZSH you may have problems with the square braquets, in such a case you can do:
   
   .. code-block:: console
      
      pip3 install -e MITIM-fusion\[pyqt\]

.. dropdown:: ``ERROR: Wheel`` error in pip:

   Make sure you are getting the fresh packages, by using the ``--no-cache`` option:
   
   .. code-block:: console
      
      pip3 install -e MITIM-fusion\[pyqt\] --no-cache-dir

Issues during MITIM tests
-------------------------

.. dropdown:: TGLF simulations do not run

   Make sure you do have the full setup to run TGLF available in your machine upon logging-in.
   For example:

   .. code-block:: console
      
      export GACODE_PLATFORM=OSX_MONTEREY
      export GACODE_ROOT=/Users/$USER/gacode
      . $GACODE_ROOT/shared/bin/gacode_setup
      . ${GACODE_ROOT}/platform/env/env.${GACODE_PLATFORM}

      \# Add also modules that are required to run MPI instances in your machine

   If you still have problems with MITIM execution of TGLF and you have checked that by manually logging-in to the machine you can run TGLF,
   then it is possible that you have print or echo statements in your ``.bashrc`` or ``.zshrc`` files.
   Please remove them or add the following:

   .. code-block:: console
      
      ! [ -z "$PS1" ] && echo "Example echo statement that only runs in interactive shells"

.. dropdown:: ``AuthenticationException: Authentication failed`` error when connecting to server through tunnel:

   Make sure you that, if you have keys, you have added them to authorized_keys in both server and tunnel machines.
   
