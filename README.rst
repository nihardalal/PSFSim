|badge1| |badge2|

.. |badge1| image:: https://codecov.io/gh/Roman-HLIS-Cosmology-PIT/PSFSim/graph/badge.svg?token=kI7xCYckoa :target: https://codecov.io/gh/Roman-HLIS-Cosmology-PIT/PSFSim

.. |badge2| image:: https://github.com/Roman-HLIS-Cosmology-PIT/PSFSim/actions/workflows/smoke-test.yml/badge.svg

PSFSim
######

Repository for simulations for testing point spread function measurements for the Roman Space Telescope. In development. Preliminary documentation can be found `here`_ via ReadTheDocs.

.. _here: https://psfsim.readthedocs.io/en/latest/index.html

Added features include position dependent pupil mask, detector effects, Zernike path differences, and more!

Instructions for making a PSF model using ``PSFSim`` are `here <docs/usage.rst>`_.

Most code lives in ``PSFObject.py``, but it may also be useful to look at ``opticsPSF.py`` and ``filter_detector_properties.py``.

Additional information:

- `Model description <docs/model.rst>`_.
- `Coordinates in PSFSim <docs/coordinates.rst>`_.

Written by Nihar Dalal, Chris Hirata, Charuhas Shiveshwarkar, Elle Moore, Katherine Laliotis, Anthony Harbo Torres, Chun-Hao To, and David Kuhtenia.

Data files:

- The associated data files ``wim_zernikes_cycle9.csv.gz`` and functions in ``wfi_data.py`` are based on optical models provided by the Roman Project.

- The optical model in ``romantrace.py`` is based on RST-SYS-SPEC-0055, Revision E.
