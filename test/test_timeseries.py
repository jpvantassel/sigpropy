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
        for value in ["values", ["a", "b", "c"]]:
            self.assertRaises(TypeError,
                              sigpropy.TimeSeries._check_input,
                              name="blah",
                              values=value)
        for value in [[1, 2, 3], (1, 2, 3), [[1, 2, 3], [4, 5, 6]]]:
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
        # No pre-event delay
        dt = 0.5
        amp = [0, 1, 2, 3]
        true_time = [0., 0.5, 1., 1.5]
        test = sigpropy.TimeSeries(amp, dt)
        self.assertListEqual(test.time.tolist(), true_time)

        # With pre-event delay
        dt = 0.5
        amp = [-1, 0, 1, 0, -1]
        true_time = [-0.5, 0., 0.5, 1., 1.5]
        test = sigpropy.TimeSeries(amp, dt, delay=-0.5)
        self.assertListEqual(test.time.tolist(), true_time)

        # 2d amp
        dt = 1
        amp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        test = sigpropy.TimeSeries(amp, dt)
        test.split(3)
        expected = np.array([[0, 1, 2, 3],
                             [3, 4, 5, 6],
                             [6, 7, 8, 9]])
        self.assertArrayEqual(expected, test.time)
        self.assertEqual(test.time.size, test.amp.size)

        # 2d amp
        dt = 1
        amp = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        test = sigpropy.TimeSeries(amp, dt)
        test.split(4)
        expected = np.array([[0, 1, 2, 3, 4],
                             [4, 5, 6, 7, 8]])
        self.assertArrayEqual(expected, test.time)
        self.assertEqual(test.time.size, test.amp.size)

    def test_split(self):
        amp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dt = 1
        thist = sigpropy.TimeSeries(amp, dt)
        thist.split(2)
        expected = np.array([[1, 2, 3],
                             [3, 4, 5],
                             [5, 6, 7],
                             [7, 8, 9]])
        self.assertArrayEqual(expected, thist.amp)

        amp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dt = 0.5
        thist = sigpropy.TimeSeries(amp, dt)
        thist.split(1)
        expected = np.array([[1, 2, 3],
                             [3, 4, 5],
                             [5, 6, 7],
                             [7, 8, 9]])
        self.assertArrayEqual(expected, thist.amp)

        amp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dt = 1
        thist = sigpropy.TimeSeries(amp, dt)
        thist.split(4)
        expected = np.array([[1, 2, 3, 4, 5],
                             [5, 6, 7, 8, 9]])
        self.assertArrayEqual(expected, thist.amp)

    def test_cosine_taper(self):
        # 0% Window - (i.e., no taper)
        amp = np.ones(10)
        dt = 1
        thist = sigpropy.TimeSeries(amp, dt)
        thist.cosine_taper(0)
        expected = amp
        self.assertArrayAlmostEqual(expected, thist.amp, places=6)

        # 50% window
        amp = np.ones(10)
        dt = 1
        thist = sigpropy.TimeSeries(amp, dt)
        thist.cosine_taper(0.5)
        expected = np.array([0.000000000000000e+00, 4.131759111665348e-01,
                             9.698463103929542e-01, 1.000000000000000e+00,
                             1.000000000000000e+00, 1.000000000000000e+00,
                             1.000000000000000e+00, 9.698463103929542e-01,
                             4.131759111665348e-01, 0.000000000000000e+00])
        self.assertArrayAlmostEqual(expected, thist.amp, places=6)

        # 100% Window
        amp = np.ones(10)
        dt = 1
        thist = sigpropy.TimeSeries(amp, dt)
        thist.cosine_taper(1)
        expected = np.array([0.000000000000000e+00, 1.169777784405110e-01,
               4.131759111665348e-01, 7.499999999999999e-01,
               9.698463103929542e-01, 9.698463103929542e-01,
               7.500000000000002e-01, 4.131759111665350e-01,
               1.169777784405111e-01, 0.000000000000000e+00])
        self.assertArrayAlmostEqual(expected, thist.amp, places=6)

    def test_from_trace(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = obspy.read(self.full_path+"data/vuws/1.dat")[0]
        tseries = sigpropy.TimeSeries.from_trace(trace, delay=-0.5)
        self.assertListEqual(tseries.amp.tolist(), trace.data.tolist())
        self.assertEqual(tseries.dt, trace.stats.delta)
        self.assertEqual(tseries._nstack, int(trace.stats.seg2.STACK))
        self.assertEqual(tseries.delay, float(trace.stats.seg2.DELAY))

    def test_trim(self):
        # Standard
        thist = sigpropy.TimeSeries(amplitude=[0, 1, 2, 3, 4],
                                    dt=0.5)
        thist.trim(0, 1)
        self.assertListEqual(thist.amp.tolist(), [0, 1, 2])
        self.assertEqual(thist.n_samples, 3)
        self.assertEqual(min(thist.time), 0)
        self.assertEqual(max(thist.time), 1)

        # With pre-trigger delay
        thist = sigpropy.TimeSeries(amplitude=[0, 1, 2, 3, 4, 5, 6],
                                    dt=0.25,
                                    delay=-.5)
        # Remove part of pre-trigger
        thist.trim(-0.25, 0.25)
        self.assertEqual(thist.n_samples, 3)
        self.assertEqual(thist.delay, -0.25)
        self.assertEqual(min(thist.time), -0.25)
        self.assertEqual(max(thist.time), 0.25)
        # Remove all of pre-trigger
        thist.trim(0, 0.25)
        self.assertEqual(thist.n_samples, 2)
        self.assertEqual(thist.delay, 0)
        self.assertEqual(min(thist.time), 0.)
        self.assertEqual(max(thist.time), 0.25)

    def test_detrend(self):
        # 1d amp
        signal = np.array([0., .2, 0., -.2]*5)
        trend = np.arange(0, 20, 1)
        amp = signal + trend
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        tseries.detrend()
        self.assertArrayAlmostEqual(signal, tseries.amp, delta=0.03)

        # 2d amp
        signal = np.array([0., .2, 0., -.2]*5)
        trend = np.arange(0, 20, 1)
        amp = signal + trend
        amp = np.vstack((amp, amp))
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        tseries.detrend()
        for row in tseries.amp:
            self.assertArrayAlmostEqual(signal,  row, delta=0.03)

    def test_to_and_from_dict(self):
        # 1d amp
        amplitude = np.array([1,2,3,4])
        dt = 1
        expected = sigpropy.TimeSeries(amplitude, dt)
        dict_repr = expected.to_dict()
        returned = sigpropy.TimeSeries.from_dict(dict_repr)
        self.assertEqual(expected.dt, returned.dt)
        self.assertArrayEqual(expected.amp, returned.amp)

    def test_to_and_from_json(self):
        # 1d amp
        amplitude = np.array([1,2,3,4])
        dt = 1
        expected = sigpropy.TimeSeries(amplitude, dt)
        json_repr = expected.to_json()
        returned = sigpropy.TimeSeries.from_json(json_repr)
        self.assertEqual(expected.dt, returned.dt)
        self.assertArrayEqual(expected.amp, returned.amp)

if __name__ == '__main__':
    unittest.main()
