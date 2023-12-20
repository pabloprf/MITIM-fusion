Frequently Asked Questions
==========================

.. contents::
   :local:
   :depth: 1

Issues during MITIM setup
-------------------------

.. dropdown:: ``pyqt`` fails to install 

   Some ``pyqt`` installation problems can be fixed with upgrading the ``pip`` package manager:
   
   .. code-block:: console
      
      pip3 install pip --upgrade

.. dropdown:: Square brackets not understood by shell

   If you are using ZSH you may have problems with the square braquets, in such a case you can do:
   
   .. code-block:: console
      
      pip3 install -e $MITIM_PATH\[pyqt\]

Issues during MITIM tests
-------------------------

Nothing here yet.

Issues during PORTALS simulations
---------------------------------

Nothing here yet.
