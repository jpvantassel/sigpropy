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

"""Tests for FourierTransformSuite class."""

import sigpropy
import obspy
import numpy as np
import pandas as pd
import warnings
from testtools import unittest, TestCase, get_full_path


class Test_FourierTransformSuite(TestCase):

    def setUp(self):
        self.full_path = get_full_path(__file__)

    def test_init(self):
        frequency = np.array([1, 2, 3])
        amplitude = np.array([[1, 2, 3], [4, 5, 6]])
        fft_suite = sigpropy.FourierTransformSuite(amplitude, frequency)

        self.assertArrayEqual(frequency, fft_suite.frequency)
        self.assertArrayEqual(amplitude, fft_suite.amplitude)

    def test_check_input(self):
        amp = [[1, 2], [3, 4]]
        frq = [1, 2]

        # TypeError - 2D Frequeny
        self.assertRaises(TypeError,
                          sigpropy.FourierTransformSuite._check_input,
                          amplitude=amp, frequency=[[4, 5], [6, 7]], fnyq=2)

        # TypeError - 1D Amplitude
        self.assertRaises(TypeError,
                          sigpropy.FourierTransformSuite._check_input,
                          amplitude=[1, 2], frequency=frq, fnyq=2)

        # ValueError - fnyq <= 0
        for fnyq in [-1, 0]:
            self.assertRaises(ValueError,
                              sigpropy.FourierTransformSuite._check_input,
                              amplitude=amp, frequency=frq, fnyq=fnyq)

    def test_konno_and_ohmachi(self):
        amp = [3.0+0.0*1j, 0.0+0.0*1j, 0.0+0.0*1j, -1.0+1.0*1j,
               0.0+0.0*1j, 0.0+0.0*1j, -1.0+0.0*1j, 0.0+0.0*1j,
               0.0+0.0*1j, -1.0+-1.0*1j, 0.0+0.0*1j, 0.0+0.0*1j]
        frq = [0.0, 0.08333333333333333, 0.16666666666666666,
               0.25, 0.3333333333333333, 0.41666666666666663,
               0.5, 0.41666666666666663, 0.3333333333333333,
               0.25, 0.16666666666666666, 0.08333333333333333]

        amp = np.array([amp, amp])
        fseries = sigpropy.FourierTransformSuite(amp, frq)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fseries.smooth_konno_ohmachi(40)

        expected = np.array([3.000000000000000, 3.50436561989e-08,
                             0.000129672263813, 1.412146598800000,
                             0.001963869435750, 1.71009155425e-05,
                             0.999819070332000, 1.71009155425e-05,
                             0.001963869435750, 1.412146598800000,
                             0.000129672263813, 3.50436561989e-08])
        for row in range(amp.shape[0]):
            self.assertArrayAlmostEqual(expected, fseries.amp[row])

    def test_smooth_konno_ohmachi_fast(self):

        def smooth(amp, windows):
            smoothed_amp = np.empty_like(amp)
            # Transpose b/c pandas uses row major.
            for cindex, window in enumerate(windows.T):
                smoothed_amp[:, cindex] = np.sum(
                    window*amp, axis=1) / np.sum(window)
            return smoothed_amp

        def load_and_run(amp, b, fname):
            dat = pd.read_csv(fname)

            frq = dat["f"].values
            fcs = dat["fc"].values
            windows = dat.iloc[:, 2:].values

            fseries = sigpropy.FourierTransformSuite(amp, frq)
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
            amp = np.vstack((amp, amp))
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
        new_amp = np.vstack((amp, amp, amp, amp, amp))
        fft = sigpropy.FourierTransformSuite(new_amp, frq)
        fc = np.logspace(np.log10(0.3), np.log10(40), 2048)
        fft.smooth_konno_ohmachi_fast(fc, 40)
        self.assertArrayEqual(fc, fft.frq)

    def test_resample(self):
        frq = [0, 1, 2, 3, 4, 5]
        amp = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
        fseries = sigpropy.FourierTransformSuite(amp, frq)

        known_frq = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        known_vals = np.array([[0.5, 1.5, 2.5, 3.5, 4.5],
                               [0.5, 1.5, 2.5, 3.5, 4.5]])

        fseries.resample(minf=0.5, maxf=4.5, nf=5,
                         res_type='linear', inplace=True)

        for expected, returned in zip(known_frq, fseries.frq):
            self.assertArrayAlmostEqual(expected, returned, places=1)
        for expected, returned in zip(known_vals, fseries.amp):
            self.assertArrayAlmostEqual(expected, returned, places=1)

    def test_from_timeseries(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fname = "data/a2/UT.STN11.A2_C50.miniseed"
            trace = obspy.read(self.full_path+fname)[0]

        wtseries = sigpropy.WindowedTimeSeries.from_trace(trace, 120)
        fsuite = sigpropy.FourierTransformSuite.from_timeseries(wtseries)

        returned = fsuite.amp.shape
        expected = (wtseries.nwindows, fsuite.frequency.size)
        self.assertTupleEqual(expected, returned)

        # Fail with TimeSeries
        self.assertRaises(TypeError,
                          sigpropy.FourierTransformSuite.from_timeseries,
                          sigpropy.TimeSeries.from_trace(trace))


if __name__ == "__main__":
    unittest.main()
