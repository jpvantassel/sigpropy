# This file is part of SigProPy, a Python package for digital signal
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

"""Tests for FourierTransform class."""

import numpy as np
import pandas as pd

import sigpropy
from testtools import unittest, TestCase, get_full_path


class Test_FourierTransform(TestCase):

    def setUp(self):
        self.full_path = get_full_path(__file__)

    def test_init(self):
        frq = np.array([0.00000000000000000, 0.090909090909090910,
                        0.18181818181818182, 0.272727272727272700,
                        0.36363636363636365, 0.454545454545454600,
                        -0.4545454545454546, -0.36363636363636365,
                        -0.2727272727272727, -0.18181818181818182,
                        -0.09090909090909091])
        amp = np.array([25.0+0.0*1j,
                        -11.843537519677056+-3.477576385886737*1j,
                        0.22844587066117938+0.14681324646918337*1j,
                        -0.9486905697966428+-1.0948472814948405*1j,
                        0.1467467171062613+0.3213304885841657*1j,
                        -0.08296449829374097+-0.5770307602665046*1j,
                        -0.08296449829374097+0.5770307602665046*1j,
                        0.1467467171062613+-0.3213304885841657*1j,
                        -0.9486905697966428+1.0948472814948405*1j,
                        0.22844587066117938+-0.14681324646918337*1j,
                        -11.843537519677056+3.477576385886737*1j])
        fseries = sigpropy.FourierTransform(amp, frq)
        self.assertArrayEqual(amp, fseries.amplitude)
        self.assertArrayEqual(frq, fseries.frequency)

    def test_from_timeseries(self):
        amp = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
        dt = 1
        mythist = sigpropy.TimeSeries(amp, dt)
        myfft = sigpropy.FourierTransform.from_timeseries(mythist)
        true_frq = np.array([0.00000000000000000, 0.09090909090909091,
                             0.18181818181818182, 0.2727272727272727,
                             0.36363636363636365, 0.4545454545454546])
        true_amp = np.array([25.0+0.0*1j,
                             -11.843537519677056+-3.477576385886737*1j,
                             0.22844587066117938+0.14681324646918337*1j,
                             -0.9486905697966428+-1.0948472814948405*1j,
                             0.1467467171062613+0.3213304885841657*1j,
                             -0.08296449829374097+-0.5770307602665046*1j])
        true_amp *= 2/len(amp)
        for expected, returned in [(true_frq, myfft.frq), (true_amp, myfft.amp)]:
            self.assertArrayAlmostEqual(expected, returned)

    def test_smooth_konno_ohmachi(self):
        amp = [3.0+0.0*1j, 0.0+0.0*1j, 0.0+0.0*1j, -1.0+1.0*1j,
               0.0+0.0*1j, 0.0+0.0*1j, -1.0+0.0*1j, 0.0+0.0*1j,
               0.0+0.0*1j, -1.0+-1.0*1j, 0.0+0.0*1j, 0.0+0.0*1j]
        frq = [0.0, 0.08333333333333333, 0.16666666666666666,
               0.25, 0.3333333333333333, 0.41666666666666663,
               0.5, 0.41666666666666663, 0.3333333333333333,
               0.25, 0.16666666666666666, 0.08333333333333333]

        fseries = sigpropy.FourierTransform(amp, frq)
        fseries.smooth_konno_ohmachi(40)
        expected = [3.000000000000000, 3.50436561989e-08,
                    0.000129672263813, 1.412146598800000,
                    0.001963869435750, 1.71009155425e-05,
                    0.999819070332000, 1.71009155425e-05,
                    0.001963869435750, 1.412146598800000,
                    0.000129672263813, 3.50436561989e-08]
        self.assertListAlmostEqual(expected, fseries.amp.tolist())

    def test_smooth_konno_ohmachi_fast(self):

        def smooth(amp, windows):
            smoothed_amp = np.empty_like(amp)
            # Transpose b/c pandas uses row major.
            for cindex, window in enumerate(windows.T):
                smoothed_amp[cindex] = np.sum(window*amp) / np.sum(window)
            return smoothed_amp

        def load_and_run(amp, b, fname):
            dat = pd.read_csv(fname)

            frq = dat["f"].values
            fcs = dat["fc"].values
            windows = dat.iloc[:, 2:].values

            fseries = sigpropy.FourierTransform(amp, frq)
            fseries.smooth_konno_ohmachi_fast(frequencies=fcs, bandwidth=b)

            returned = fseries.amp
            expected = smooth(amp, windows)
            self.assertArrayAlmostEqual(expected, returned, places=2)

        amps = [np.array([0., 1.4]*8),
                np.array([1.2, 1.]*8),
                np.array([1.2, 0.7]*8)]

        bs = [20, 40, 60]
        fnames = ["ko_b20.csv", "ko_b40.csv", "ko_b60.csv"]

        for amp in amps:
            for b, fname in zip(bs, fnames):
                load_and_run(amp, b, self.full_path+"data/ko/"+fname)

    def test_ko_nan(self):
        with open(self.full_path + "data/ko/ko_nan.csv", "r") as f:
            lines = f.read().splitlines()
        frq, amp = [], []
        for line in lines:
            f, a = line.split(",")
            frq.append(float(f))
            amp.append(float(a))
        fft = sigpropy.FourierTransform(amp, frq)
        fc = np.logspace(np.log10(0.3), np.log10(40), 2048)
        fft.smooth_konno_ohmachi_fast(fc, 40)
        self.assertArrayEqual(fc, fft.frq)

    def test_resample(self):
        frq = [0, 1, 2, 3, 4, 5]
        amp = [0, 1, 2, 3, 4, 5]
        fseries = sigpropy.FourierTransform(amp, frq)

        # inplace = True
        expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        fseries.resample(minf=0.5, maxf=4.5, nf=5,
                         res_type='linear', inplace=True)
        for attr in ["frq", "amp"]:
            self.assertArrayEqual(expected, getattr(fseries, attr))

        # inplace = False
        expected = np.geomspace(0.5, 4.5, 5)
        returneds = fseries.resample(minf=0.5, maxf=4.5, nf=5,
                                     res_type='log', inplace=False)
        for returned in returneds:
            self.assertArrayEqual(expected, returned)

        self.assertRaises(ValueError, fseries.resample, minf=4, maxf=1, nf=5)
        self.assertRaises(TypeError, fseries.resample, minf=1, maxf=4, nf=5.1)
        self.assertRaises(ValueError, fseries.resample, minf=1, maxf=4, nf=-2)
        self.assertRaises(ValueError, fseries.resample, minf=0.1, maxf=4, nf=5)
        self.assertRaises(ValueError, fseries.resample, minf=1, maxf=7, nf=5)
        self.assertRaises(NotImplementedError, fseries.resample, minf=1,
                          maxf=4, nf=5, res_type="spline")

    def test_properties(self):
        frq = np.array([1, 2, 3, 4, 5])
        amp = np.array([0+1j, 1+2j, 4+0j, 2j, 5])
        fseries = sigpropy.FourierTransform(amp, frq)

        # .amplitude
        self.assertArrayEqual(amp, fseries.amplitude)

        # .mag
        returned = fseries.mag
        expected = np.array([1, np.sqrt(5), 4, 2, 5])
        self.assertArrayEqual(expected, returned)

        # .imag
        returned = fseries.imag
        expected = np.array([1, 2, 0, 2, 0])
        self.assertArrayEqual(expected, returned)

        # .real
        returned = fseries.real
        expected = np.array([0, 1, 4, 0, 5])
        self.assertArrayEqual(expected, returned)

        # .phase
        returned = fseries.phase
        expected = np.arctan2(fseries.imag, fseries.real)
        self.assertArrayEqual(expected, returned)


if __name__ == "__main__":
    unittest.main()
