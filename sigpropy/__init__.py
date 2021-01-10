# This file is part of sigpropy, a Python package for digital signal
# processing.
# Copyright (C) 2019-2020 Joseph P. Vantassel (jvantassel@utexas.edu)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https: //www.gnu.org/licenses/>.

"""Import classes from each module into the sigpropy package."""

import logging

from .timeseries import TimeSeries
from .windowedtimeseries import WindowedTimeSeries
from .fouriertransform import FourierTransform
from .fouriertransformsuite import FourierTransformSuite
from .meta import __version__

logging.getLogger('swprocess').addHandler(logging.NullHandler())