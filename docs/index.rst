.. SigProPy documentation master file, created by sphinx-quickstart on
   Fri Nov  8 09:52:40 2019. You can adapt this file completely to your
   liking, but it should at least contain the root `toctree` directive.

SigProPy Documentation
======================

Summary
-------

`sigpropy` is a Python package for digital signal processing. It includes two
main class definitions, `TimeSeries` and `FourierTransform`. These classes
include methods to perform common signal processing techniques (e.g., trimming
and resampling) and properties to make using them readable and intuitive.

This package and the classes therein are being used in several other
Python projects, some of which have been released publically and others are
still in the development stage, so if you do not see a feature you would like
it may very well be under development and released in the near future. To be
notified of future releases, you can either `watch` the repository on
`Github <https://github.com/jpvantassel/sigpropy>`_
`Subscribe to releases` on the
`Python Package Index (PyPI) <https://pypi.org/project/sigpropy/>`_.

License Information
-------------------

   Copyright (C) 2019-2020 Joseph P. Vantassel (jvantassel@utexas.edu)

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https: //www.gnu.org/licenses/>.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

TimeSeries Class
================
.. automodule:: sigpropy.timeseries
   :attributes:
   :members:
   :noindex:

.. autoclass:: sigpropy.TimeSeries
   :members:

   .. automethod:: __init__

FourierTransform Class
======================
.. automodule:: sigpropy.fouriertransform
   :attributes:
   :members:
   :noindex:

.. autoclass:: sigpropy.FourierTransform
   :members:

   .. automethod:: __init__

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`
