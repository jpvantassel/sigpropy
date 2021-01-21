# This file is part of sigpropy, a Python package for signal processing.
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

import warnings

import numpy as np
import pandas as pd
import obspy

import sigpropy
from testtools import unittest, TestCase, get_full_path


class Test_FourierTransform(TestCase):

    def setUp(self):
        self.full_path = get_full_path(__file__)

    def test_fft(self):
        dt = 0.5
        amplitude = np.array([[[1., 2, 3]]])
        self.assertRaises(
            TypeError, sigpropy.FourierTransform.fft, amplitude, dt)

    def test_init(self):
        # 1d amplitude
        frequency = np.array([1., 2, 3])
        amplitude = np.array([4., 5, 6], dtype=np.cdouble)
        fsuite = sigpropy.FourierTransform(amplitude, frequency)

        self.assertArrayEqual(frequency, fsuite.frequency)
        self.assertArrayEqual(amplitude, fsuite.amplitude)

        # 2d amplitude
        frequency = np.array([1., 2, 3])
        amplitude = np.array([[4, 5, 6], [7, 8, 9]], dtype=np.cdouble)
        fsuite = sigpropy.FourierTransform(amplitude, frequency)

        self.assertArrayEqual(frequency, fsuite.frequency)
        self.assertArrayEqual(amplitude, fsuite.amplitude)

        # invalid amplitude, str rather than cdouble.
        frequency = [1., 2, 3]
        amplitude = ["a", "b", "c"]
        self.assertRaises(TypeError, sigpropy.FourierTransform,
                          amplitude, frequency)

        # invalid amplitude, ndim > 2.
        frequency = [1., 2, 3]
        amplitude = np.array([[[1., 2, 3]]], dtype=np.cdouble)
        self.assertRaises(TypeError, sigpropy.FourierTransform,
                          amplitude, frequency)

        # invalid frequency, str rather than cdouble.
        frequency = ["a", "b", "c"]
        amplitude = [1., 2, 3]
        self.assertRaises(TypeError, sigpropy.FourierTransform,
                          amplitude, frequency)

        # invalid frequency, ndim > 1.
        frequency = np.array([[[1., 2, 3]]])
        amplitude = [1., 2, 3]
        self.assertRaises(TypeError, sigpropy.FourierTransform,
                          amplitude, frequency)

        # invalid fnyq (fnyq < 0).
        frequency = [1, 2, 3]
        amplitude = [4, 5, 6]
        fnyq = 0
        self.assertRaises(ValueError, sigpropy.FourierTransform,
                          amplitude, frequency, fnyq)

    def test_properties(self):
        frq = np.array([1., 2, 3, 4, 5])
        amp = np.array([0+1j, 1+2j, 4+0j, 2j, 5])
        fseries = sigpropy.FourierTransform(amp, frq)

        # .amplitude
        self.assertArrayEqual(amp, fseries.amplitude)

        # .amp Deprecated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertArrayEqual(amp, fseries.amp)

        # .frequency
        self.assertArrayEqual(frq, fseries.frequency)

        # .frq Deprecated
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertArrayEqual(frq, fseries.frq)

        # .mag
        returned = fseries.mag
        expected = np.array([1., np.sqrt(5), 4, 2, 5])
        self.assertArrayEqual(expected, returned)

        # .imag
        returned = fseries.imag
        expected = np.array([1., 2, 0, 2, 0])
        self.assertArrayEqual(expected, returned)

        # .real
        returned = fseries.real
        expected = np.array([0., 1, 4, 0, 5])
        self.assertArrayEqual(expected, returned)

        # .phase
        returned = fseries.phase
        expected = np.arctan2(fseries.imag, fseries.real)
        self.assertArrayEqual(expected, returned)

    def test_konno_and_ohmachi(self):
        amp = [3.0+0.0*1j, 0.0+0.0*1j, 0.0+0.0*1j, -1.0+1.0*1j,
               0.0+0.0*1j, 0.0+0.0*1j, -1.0+0.0*1j, 0.0+0.0*1j,
               0.0+0.0*1j, -1.0+-1.0*1j, 0.0+0.0*1j, 0.0+0.0*1j]
        frq = [0.0, 0.08333333333333333, 0.16666666666666666,
               0.25, 0.3333333333333333, 0.41666666666666663,
               0.5, 0.41666666666666663, 0.3333333333333333,
               0.25, 0.16666666666666666, 0.08333333333333333]

        amp = np.array([amp, amp])
        fseries = sigpropy.FourierTransform(amp, frq)

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
            self.assertArrayAlmostEqual(expected, fseries.amplitude[row])

        # Old example.
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
        self.assertListAlmostEqual(expected, fseries.amplitude.tolist())

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

            fseries = sigpropy.FourierTransform(amp, frq)
            fseries.smooth_konno_ohmachi_fast(frequencies=fcs, bandwidth=b)

            returned = fseries.amplitude
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
        fft = sigpropy.FourierTransform(amp, frq)
        fc = np.logspace(np.log10(0.3), np.log10(40), 2048)

        fft.smooth_konno_ohmachi_fast(fc, 40)
        self.assertArrayEqual(fc, fft.frequency)

    def test_resample(self):
        frq = [0, 1, 2, 3, 4, 5]
        amp = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]])
        fseries = sigpropy.FourierTransform(amp, frq)

        known_frq = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        known_vals = np.array([[0.5, 1.5, 2.5, 3.5, 4.5],
                               [0.5, 1.5, 2.5, 3.5, 4.5]])

        fseries.resample(minf=0.5, maxf=4.5, nf=5,
                         res_type='linear', inplace=True)

        for expected, returned in zip(known_frq, fseries.frequency):
            self.assertArrayAlmostEqual(expected, returned, places=1)
        for expected, returned in zip(known_vals, fseries.amplitude):
            self.assertArrayAlmostEqual(expected, returned, places=1)

        # Old example.
        frq = [0, 1, 2, 3, 4, 5]
        amp = [0, 1, 2, 3, 4, 5]
        fseries = sigpropy.FourierTransform(amp, frq)

        # inplace = True
        expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        fseries.resample(minf=0.5, maxf=4.5, nf=5,
                         res_type='linear', inplace=True)
        for attr in ["frequency", "amplitude"]:
            self.assertArrayEqual(expected, getattr(fseries, attr).real)

        # inplace = False
        expected = np.geomspace(0.5, 4.5, 5)
        returneds = fseries.resample(minf=0.5, maxf=4.5, nf=5,
                                     res_type='log', inplace=False)
        for returned in returneds:
            self.assertArrayEqual(expected, returned.real)

        self.assertRaises(ValueError, fseries.resample, minf=4, maxf=1, nf=5)
        self.assertRaises(TypeError, fseries.resample, minf=1, maxf=4, nf=5.1)
        self.assertRaises(ValueError, fseries.resample, minf=1, maxf=4, nf=-2)
        self.assertRaises(ValueError, fseries.resample, minf=0.1, maxf=4, nf=5)
        self.assertRaises(ValueError, fseries.resample, minf=1, maxf=7, nf=5)
        self.assertRaises(NotImplementedError, fseries.resample, minf=1,
                          maxf=4, nf=5, res_type="spline")

    def test_from_timeseries(self):
        # TODO (jpv): Replace these tests with something more meaningful.
        # Early example.
        amplitude = [0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0]
        dt = 1
        tseries = sigpropy.TimeSeries(amplitude, dt)
        fft = sigpropy.FourierTransform.from_timeseries(tseries)
        true_frq = np.array([0.00000000000000000, 0.09090909090909091,
                             0.18181818181818182, 0.2727272727272727,
                             0.36363636363636365, 0.4545454545454546])
        true_amp = np.array([25.0+0.0*1j,
                             -11.843537519677056+-3.477576385886737*1j,
                             0.22844587066117938+0.14681324646918337*1j,
                             -0.9486905697966428+-1.0948472814948405*1j,
                             0.1467467171062613+0.3213304885841657*1j,
                             -0.08296449829374097+-0.5770307602665046*1j])
        true_amp *= 2/len(amplitude)
        for expected, returned in [(true_frq, fft.frequency), (true_amp, fft.amplitude)]:
            self.assertArrayAlmostEqual(expected, returned)

        # Load and transform ambient noise record.
        fname = "data/a2/UT.STN11.A2_C50.miniseed"
        trace = obspy.read(self.full_path+fname)[0]

        tseries = sigpropy.TimeSeries.from_trace(trace)
        tseries.split(windowlength=120)
        fsuite = sigpropy.FourierTransform.from_timeseries(tseries)

        returned = fsuite.amplitude.shape
        expected = (tseries.nseries, fsuite.frequency.size)
        self.assertTupleEqual(expected, returned)

        # Bad timeseries.
        self.assertRaises(TypeError,
                          sigpropy.FourierTransform.from_timeseries,
                          "bad timeseries")

    def test_str_and_repr(self):
        frequency = [1., 2, 3]
        amplitude = [4., 5, 6]
        fseries = sigpropy.FourierTransform(amplitude, frequency)

        # str
        expected = f"FourierTransform of shape (3,) at {id(fseries)}"
        self.assertEqual(expected, fseries.__str__())

        # repr
        expected = f"FourierTransform(amplitude=np.array([ 4.+0.j,  5.+0.j,  6.+0.j]), frequency=np.array([ 1.,  2.,  3.]), fnyq=3.0)"
        self.assertEqual(expected, fseries.__repr__())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
