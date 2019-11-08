# This file is part of SigProPy a module for digital signal processing
# in python.
# Copyright (C) 2019 Joseph P. Vantassel (jvantassel@utexas.edu)
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

"""Tests for FourierTransform class."""

import unittest
import sigpropy
import obspy
import numpy as np
import matplotlib.pyplot as plt


class TestFourierTransform(unittest.TestCase):
    # def test_init(self):
    #     frq = np.array([0.0,
    #                     0.09090909090909091,
    #                     0.18181818181818182,
    #                     0.2727272727272727,
    #                     0.36363636363636365,
    #                     0.4545454545454546,
    #                     -0.4545454545454546,
    #                     -0.36363636363636365,
    #                     -0.2727272727272727,
    #                     -0.18181818181818182,
    #                     -0.09090909090909091])
    #     amp = np.array([25.0+0.0*1j,
    #                     -11.843537519677056+-3.477576385886737*1j,
    #                     0.22844587066117938+0.14681324646918337*1j,
    #                     -0.9486905697966428+-1.0948472814948405*1j,
    #                     0.1467467171062613+0.3213304885841657*1j,
    #                     -0.08296449829374097+-0.5770307602665046*1j,
    #                     -0.08296449829374097+0.5770307602665046*1j,
    #                     0.1467467171062613+-0.3213304885841657*1j,
    #                     -0.9486905697966428+1.0948472814948405*1j,
    #                     0.22844587066117938+-0.14681324646918337*1j,
    #                     -11.843537519677056+3.477576385886737*1j])
    #     sigpropy.FourierTransform()
    # TODO (jpv): Remove inconsistency between freq and amp.

    def test_smooth_konno_ohmachi(self):
        traces = obspy.read("test/data/a2/UT.STN11.A2_C50.miniseed")
        mythist = sigpropy.TimeSeries.from_trace(traces[0])
        mythist.trim(0, 100)
        myfft = sigpropy.FourierTransform.from_timeseries(mythist)
        # plt.plot(myfft.frq[1:], myfft.mag[1:], '-k')
        myfft.smooth_konno_ohmachi()
        # plt.plot(myfft.frq[1:], myfft.mag[1:], '--c')
        # plt.xscale('log')
        # plt.show()


# myfft = sigpropy.FourierTransform.from_timeseries(mythist)
#         # plt.plot(myfft.frq, myfft.mag)
#         myfft.smooth_konno_ohmachi()
#         # plt.plot(myfft.frq, myfft.mag)
#         # plt.xscale('log')
#         # plt.show()

if __name__ == "__main__":
    unittest.main()
