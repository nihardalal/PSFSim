.. image:: https://codecov.io/gh/nihardalal/PSFSim/graph/badge.svg?token=O76KO3TQA2 
 :target: https://codecov.io/gh/nihardalal/PSFSim

PSFSim
######

Repository for simulations for testing point spread function measurements for the Roman Space Telescope. In development. Preliminary documentation can be found `here`_ via ReadTheDocs.

.. _here: https://psfsim.readthedocs.io/en/latest/index.html

Added features include position dependent pupil mask, detector effects, Zernike path differences, and more!

Most code lives in ``PSFObject.py``, but it may also be useful to look at ``opticsPSF.py`` and ``filter_detector_properties.py``.

Written by Nihar Dalal, Charuhas Shiveshwarkar, and Chris Hirata

Data files:

- The associated data files ``wim_zernikes_cycle9.csv.gz`` and functions in ``wfi_data.py`` are based on optical models provided by the Roman Project.

- The optical model in ``romantrace.py`` is based on RST-SYS-SPEC-0055, Revision E.
