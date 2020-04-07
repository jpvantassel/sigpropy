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

"""Tests for TimeSeries class."""

import sigpropy
import obspy
import numpy as np
import warnings
import logging
from testtools import get_full_path, unittest, TestCase
logging.basicConfig(level=logging.WARNING)


class Test_TimeSeries(TestCase):

    def setUp(self):
        self.full_path = get_full_path(__file__)

    def test_check(self):
        for value in ["values", ["a", "b", "c"], [[1,2,3]]]:
            self.assertRaises(TypeError,
                              sigpropy.TimeSeries._check_input,
                              name="blah",
                              values=value)
        for value in [[1, 2, 3], (1, 2, 3)]:
            value = sigpropy.TimeSeries._check_input(name="blah", values=value)
            self.assertTrue(isinstance(value, np.ndarray))

    def test_init(self):
        dt = 1
        amp = [0, 1, 0, -1]
        test = sigpropy.TimeSeries(amp, dt)
        self.assertListEqual(amp, test.amp.tolist())
        self.assertEqual(dt, test.dt)

        amp = np.array(amp)
        test = sigpropy.TimeSeries(amp, dt)
        self.assertArrayEqual(amp, test.amp)

    def test_time(self):
        dt = 0.5
        amp = [0, 1, 2, 3]
        expected = np.array([0., 0.5, 1., 1.5])
        test = sigpropy.TimeSeries(amp, dt)
        returned = test.time
        self.assertArrayEqual(expected, returned)

    def test_split(self):
        amp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        self.assertRaises(DeprecationWarning, tseries.split, 2)

    def test_cosine_taper(self):
        # 0% Window - (i.e., no taper)
        amp = np.ones(10)
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        tseries.cosine_taper(0)
        returned = tseries.amp
        expected = amp
        self.assertArrayEqual(expected, returned)

        # 50% window
        amp = np.ones(10)
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        tseries.cosine_taper(0.5)
        returned = tseries.amp
        expected = np.array([0.000000000000000e+00, 4.131759111665348e-01,
                             9.698463103929542e-01, 1.000000000000000e+00,
                             1.000000000000000e+00, 1.000000000000000e+00,
                             1.000000000000000e+00, 9.698463103929542e-01,
                             4.131759111665348e-01, 0.000000000000000e+00])
        self.assertArrayAlmostEqual(expected, returned, places=6)

        # 100% Window
        amp = np.ones(10)
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        tseries.cosine_taper(1)
        returned = tseries.amp
        expected = np.array([0.000000000000000e+00, 1.169777784405110e-01,
               4.131759111665348e-01, 7.499999999999999e-01,
               9.698463103929542e-01, 9.698463103929542e-01,
               7.500000000000002e-01, 4.131759111665350e-01,
               1.169777784405111e-01, 0.000000000000000e+00])
        self.assertArrayAlmostEqual(expected, returned, places=6)

    def test_from_trace(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = obspy.read(self.full_path+"data/vuws/1.dat")[0]
        tseries = sigpropy.TimeSeries.from_trace(trace)
        self.assertArrayEqual(tseries.amp, trace.data)
        self.assertEqual(tseries.dt, trace.stats.delta)

    def test_trim(self):
        tseries = sigpropy.TimeSeries(amplitude=[0, 1, 2, 3, 4],
                                    dt=0.5)
        tseries.trim(0, 1)
        self.assertListEqual(tseries.amp.tolist(), [0, 1, 2])
        self.assertEqual(tseries.nsamples, 3)
        self.assertEqual(min(tseries.time), 0)
        self.assertEqual(max(tseries.time), 1)

    def test_detrend(self):
        signal = np.array([0., .2, 0., -.2]*5)
        trend = np.arange(0, 20, 1)
        amp = signal + trend
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        tseries.detrend()
        returned = tseries.amp
        expected = signal
        self.assertArrayAlmostEqual(expected, returned, delta=0.03)

    def test_to_and_from_dict(self):
        amplitude = np.array([1,2,3,4])
        dt = 1
        expected = sigpropy.TimeSeries(amplitude, dt)
        dict_repr = expected.to_dict()
        returned = sigpropy.TimeSeries.from_dict(dict_repr)
        self.assertEqual(expected.dt, returned.dt)
        self.assertArrayEqual(expected.amp, returned.amp)

    def test_to_and_from_json(self):
        amplitude = np.array([1,2,3,4])
        dt = 1
        expected = sigpropy.TimeSeries(amplitude, dt)
        json_repr = expected.to_json()
        returned = sigpropy.TimeSeries.from_json(json_repr)
        self.assertEqual(expected.dt, returned.dt)
        self.assertArrayEqual(expected.amp, returned.amp)

if __name__ == '__main__':
    unittest.main()
