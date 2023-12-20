MITIM: a toolbox for modeling tasks in plasma physics and fusion energy
=======================================================================

The **MITIM** (MIT Integrated Modeling) is a versatile and user-friendly Python library designed for *plasma physics* and *fusion energy* researchers.
Developed in 2018 by `Pablo Rodriguez-Fernandez <https://www.pablorf.com/>`_ at the MIT Plasma Science and Fusion Center, this light-weight, command-line,
object-oriented toolbox streamlines the execution and interpretation of physics models and simplifies complex optimization tasks.

MITIM stands out for its modular nature, making it particularly useful for integrating models with optimization workflows.
The toolbox has been instrumental in various high-impact research projects, such as `VITALS <https://www.tandfonline.com/doi/abs/10.1080/15361055.2017.1396166>`_
and `PORTALS <https://iopscience.iop.org/article/10.1088/1741-4326/ac64b2>`_, as well as optimization tasks during the design of the `SPARC tokamak <https://iopscience.iop.org/article/10.1088/1741-4326/ac1654>`_.
MITIM's ongoing development and maintenance by the `MFE-IM group <https://mfeim.mit.edu/>`_ at MIT ensure its relevance and utility in cutting-edge research,
with a focus on core transport, simulation and optimization.

Researchers and developers interested in contributing can find the project on GitHub at `MITIM-fusion <https://github.com/pabloprf/MITIM-fusion/>`_.
The repository welcomes contributions and provides guidelines for those looking to enhance MITIM's capabilities.

Users Agreement: :ref:`License and Contributions`

.. warning::

   The authors are not responsible for any errors or omissions, or for the results obtained from the use of this repository. All scripts and coding examples in this repository are provided "as is", with no guarantee of completeness, accuracy or of the results obtained.

   The intended use of this repository and the capabilities it provides is to accelerate the learning curve of main transport codes, specially for students and young researchers. For publication-quality results, the user is advised to understand every step behind the wheels of MITIM, and to write custom workflows and routines to test and verify results.

   The users are strongly encouraged to contribute to the code by submitting issues, requesting features or finding bugs. The Users Agreement applies to any forked version of the repository.

Overview
--------

Developed at the MIT Plasma Science and Fusion Center, MITIM emerged in 2023 as a progression from the PORTALS project (*Performance Optimization of Reactors via Training of Active Learning Surrogates*).
This evolution marks a significant enhancement in our approach to transport and optimization in plasma physics research.

MITIM's core functionality revolves around the standalone execution of codes and the nuanced interpretation of results through object-oriented Python scripts.
This enables researchers to seamlessly integrate these scripts into custom surrogate-based optimization frameworks,
significantly boosting the efficiency and effectiveness of their studies (see :ref:`Standalone Capabilities` and :ref:`Optimization Capabilities` for more details).

.. toctree::
   :caption: Contents
   :maxdepth: 2

   installation
   capabilities/standalone
   capabilities/optimization
   faq

.. note:: 
   Language enhancements and code refinements provided by the assistance of OpenAI's ChatGPT-4.