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

import warnings
import logging

import obspy
import numpy as np

import sigpropy
from testtools import get_full_path, unittest, TestCase

logging.basicConfig(level=logging.WARNING)


class Test_TimeSeries(TestCase):

    def setUp(self):
        self.full_path = get_full_path(__file__)

    def test_init(self):
        # nseries = 1
        dt = 1
        amplitude = [0, 1, 0, -1]
        test = sigpropy.TimeSeries(amplitude, dt)
        self.assertListEqual(amplitude, test.amplitude.tolist())
        self.assertEqual(dt, test.dt)

        amplitude = np.array(amplitude, dtype=np.double)
        test = sigpropy.TimeSeries(amplitude, dt)
        self.assertArrayEqual(amplitude, test.amplitude)

        # nseries = 2
        dt = 1
        amplitude = [[0, 1, 0, -1], [0, 1, 0, -1]]
        test = sigpropy.TimeSeries(amplitude, dt)
        self.assertNestedListEqual(amplitude, test.amplitude.tolist())
        self.assertEqual(dt, test.dt)

        # nseries = 3
        dt = 1
        amplitude = [[0, 1, 0, -1], [0, 1, 0, 1], [0, -1, 0, 1]]
        test = sigpropy.TimeSeries(amplitude, dt)
        self.assertNestedListEqual(amplitude, test.amplitude.tolist())
        self.assertEqual(dt, test.dt)

        amplitude = np.array(amplitude, dtype=np.double)
        test = sigpropy.TimeSeries(amplitude, dt)
        self.assertArrayEqual(amplitude, test.amplitude)

        # Invalid entries
        for amplitude in ["values", ["a", "b", "c"]]:
            self.assertRaises(TypeError,
                              sigpropy.TimeSeries,
                              amplitude=amplitude,
                              dt=1)

    def test_time(self):
        # nseries=1, short and simple.
        dt = 0.5
        amplitude = [0, 1, 2, 3]
        expected = np.array([0., 0.5, 1., 1.5])
        test = sigpropy.TimeSeries(amplitude, dt)
        returned = test.time
        self.assertArrayEqual(expected, returned)

        # nseries=1, long time series (1 day @ 200 Hz).
        dt = 0.005
        amplitude = np.arange(0, 24*3600*int(1/dt))
        test = sigpropy.TimeSeries(amplitude, dt)
        expected = amplitude*dt
        returned = test.time
        self.assertArrayEqual(expected, returned)

        # nseries=2, short and simple
        dt = 1
        amplitude = [0., 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        tseries = sigpropy.TimeSeries(amplitude, dt)
        tseries.split(windowlength=4)
        returned = tseries.time
        expected = np.array([[0., 1, 2, 3, 4],
                             [4., 5, 6, 7, 8]])
        self.assertArrayEqual(expected, returned)

        # nseries=3, short and simple.
        dt = 1
        amplitude = [0., 1, 2, 3, 4, 5, 6, 7, 8, 9]
        tseries = sigpropy.TimeSeries(amplitude, dt)
        tseries.split(windowlength=3)
        returned = tseries.time
        expected = np.array([[0., 1, 2, 3],
                             [3., 4, 5, 6],
                             [6., 7, 8, 9]])
        self.assertArrayEqual(expected, returned)

    def test_trim(self):
        # nseries = 1
        tseries = sigpropy.TimeSeries(amplitude=[0, 1, 2, 3, 4],
                                      dt=0.5)
        tseries.trim(0, 1)
        self.assertListEqual([0, 1, 2], tseries.amplitude.tolist())
        self.assertEqual(3, tseries.nsamples)
        self.assertEqual(0, min(tseries.time))
        self.assertEqual(1, max(tseries.time))

        # nseries = 3, no trim
        amplitude = np.array([[0., 1, 2, 3], [3., 4, 5, 6], [6., 7, 8, 9]])
        tseries = sigpropy.TimeSeries(amplitude=amplitude, dt=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tseries.trim(0, 9)
        expected = amplitude
        returned = tseries.amplitude
        self.assertArrayEqual(expected, returned)

        # nseries = 3, remove last window
        for end_time in [6, 7, 8]:
            amplitude = np.array([[0., 1, 2, 3], [3., 4, 5, 6], [6., 7, 8, 9]])
            tseries = sigpropy.TimeSeries(amplitude=amplitude, dt=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tseries.trim(0, end_time)
            returned = tseries.amplitude
            expected = amplitude[:-1]
            self.assertArrayEqual(expected, returned)

        # nseries = 3, leave only the middle window.
        for end_time in [6, 7, 8]:
            amplitude = np.array([[0., 1, 2, 3], [3., 4, 5, 6], [6., 7, 8, 9]])
            tseries = sigpropy.TimeSeries(amplitude=amplitude, dt=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tseries.trim(3, end_time)
            returned = tseries.amplitude
            expected = amplitude[1:2]
            self.assertArrayEqual(expected, returned)

        # nseries = 3, leave only the last window.
        amplitude = np.array([[0., 1, 2, 3], [3., 4, 5, 6], [6., 7, 8, 9]])
        tseries = sigpropy.TimeSeries(amplitude=amplitude, dt=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tseries.trim(6, 9)
        returned = tseries.amplitude
        expected = amplitude[2:3]
        self.assertArrayEqual(expected, returned)

        # nseries = 3, leave only the first window.
        for end_time in [3, 4, 5]:
            amplitude = np.array([[0., 1, 2, 3], [3., 4, 5, 6], [6., 7, 8, 9]])
            tseries = sigpropy.TimeSeries(amplitude=amplitude, dt=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tseries.trim(0, end_time)
            returned = tseries.amplitude
            expected = amplitude[0:1]
            self.assertArrayEqual(expected, returned)

    def test_split(self):
        # nseries = 1 to nseries = 2
        amplitude = [1., 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dt = 1
        tseries = sigpropy.TimeSeries(amplitude=amplitude, dt=dt)
        tseries.split(windowlength=4)
        returned = tseries.amplitude
        expected = np.array([[1., 2, 3, 4, 5],
                             [5., 6, 7, 8, 9]])
        self.assertArrayEqual(expected, returned)
        self.assertEqual(2, tseries.nseries)
        self.assertEqual(5, tseries.nsamples)

        # nseries = 1 to nseries = 3
        dt = 1
        amplitude = [0., 1, 2, 3, 4, 5, 6, 7, 8, 9]
        tseries = sigpropy.TimeSeries(amplitude=amplitude, dt=dt)
        tseries.split(windowlength=3)
        returned = tseries.amplitude
        expected = np.array([[0., 1, 2, 3],
                             [3., 4, 5, 6],
                             [6., 7, 8, 9]])
        self.assertArrayEqual(expected, returned)
        self.assertEqual(3, tseries.nseries)
        self.assertEqual(4, tseries.nsamples)

        # nseries = 1 to nseries = 4
        amplitude = [1., 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dt = 1
        tseries = sigpropy.TimeSeries(amplitude=amplitude, dt=dt)
        tseries.split(windowlength=2)
        returned = tseries.amplitude
        expected = np.array([[1., 2, 3],
                             [3., 4, 5],
                             [5., 6, 7],
                             [7., 8, 9]])
        self.assertArrayEqual(expected, returned)
        self.assertEqual(4, tseries.nseries)
        self.assertEqual(3, tseries.nsamples)

        # nseries = 3 to nseries = 4
        amplitude = np.array([[1., 2, 3, 4, 5],
                              [5., 6, 7, 8, 9],
                              [9., 0, 1, 2, 3]])
        dt = 1
        tseries = sigpropy.TimeSeries(amplitude=amplitude, dt=dt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tseries.split(windowlength=3)
        returned = tseries.amplitude
        expected = np.array([[1., 2, 3, 4],
                             [4., 5, 6, 7],
                             [7., 8, 9, 0],
                             [0., 1, 2, 3]])
        self.assertArrayEqual(expected, returned)

    def test_cosine_taper(self):
        ## nseries = 1
        # 0% Window - (i.e., no taper)
        amp = np.ones(10)
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        tseries.cosine_taper(0)
        returned = tseries.amplitude
        expected = amp
        self.assertArrayEqual(expected, returned)

        # 50% window
        amp = np.ones(10)
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        tseries.cosine_taper(0.5)
        returned = tseries.amplitude
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
        returned = tseries.amplitude
        expected = np.array([0.000000000000000e+00, 1.169777784405110e-01,
                             4.131759111665348e-01, 7.499999999999999e-01,
                             9.698463103929542e-01, 9.698463103929542e-01,
                             7.500000000000002e-01, 4.131759111665350e-01,
                             1.169777784405111e-01, 0.000000000000000e+00])
        self.assertArrayAlmostEqual(expected, returned, places=6)

        ## nseries = 3
        # 0% Window - (i.e., no taper)
        amplitude = np.ones(30)
        dt = 1
        tseries = sigpropy.TimeSeries(amplitude, dt)
        tseries.split(windowlength=9)
        tseries.cosine_taper(0)
        expected = np.ones(10)
        for returned in tseries.amplitude:
            self.assertArrayAlmostEqual(expected, returned, places=6)

        # 50% window
        amplitude = np.ones(30)
        dt = 1
        tseries = sigpropy.TimeSeries(amplitude, dt)
        tseries.split(windowlength=9)
        tseries.cosine_taper(0.5)
        expected = np.array([0.000000000000000e+00, 4.131759111665348e-01,
                             9.698463103929542e-01, 1.000000000000000e+00,
                             1.000000000000000e+00, 1.000000000000000e+00,
                             1.000000000000000e+00, 9.698463103929542e-01,
                             4.131759111665348e-01, 0.000000000000000e+00])
        for returned in tseries.amplitude:
            self.assertArrayAlmostEqual(expected, returned, places=6)

        # 100% Window
        amplitude = np.ones(30)
        dt = 1
        tseries = sigpropy.TimeSeries(amplitude, dt)
        tseries.split(windowlength=9)
        tseries.cosine_taper(1)
        expected = np.array([0.000000000000000e+00, 1.169777784405110e-01,
                             4.131759111665348e-01, 7.499999999999999e-01,
                             9.698463103929542e-01, 9.698463103929542e-01,
                             7.500000000000002e-01, 4.131759111665350e-01,
                             1.169777784405111e-01, 0.000000000000000e+00])
        for returned in tseries.amplitude:
            self.assertArrayAlmostEqual(expected, returned, places=6)

    def test_from_trace(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = obspy.read(self.full_path+"data/vuws/1.dat")[0]
        tseries = sigpropy.TimeSeries.from_trace(trace)
        self.assertEqual(trace.stats.delta, tseries.dt)
        self.assertArrayEqual(
            np.array(trace.data, dtype=np.double), tseries.amplitude)

    def test_detrend(self):
        # nseries = 1
        signal = np.array([0., .2, 0., -.2]*5)
        trend = np.arange(0, 20, 1)
        amplitude = signal + trend
        dt = 1
        tseries = sigpropy.TimeSeries(amplitude, dt)
        tseries.detrend()
        returned = tseries.amplitude
        expected = signal
        self.assertArrayAlmostEqual(expected, returned, delta=0.03)

        # nseries = 2
        signal = np.array([0., .2, 0., -.2]*5)
        trend = np.arange(0, 20, 1)
        amplitude = signal + trend
        amplitude = np.vstack((amplitude, amplitude))
        dt = 1
        tseries = sigpropy.TimeSeries(amplitude, dt)
        tseries.detrend()
        expected = signal
        for returned in tseries.amplitude:
            self.assertArrayAlmostEqual(expected, returned, delta=0.03)

    def test_to_and_from_dict(self):
        dt = 1
        amplitude = np.array([1, 2, 3, 4])
        expected = sigpropy.TimeSeries(amplitude, dt)
        dict_repr = expected.to_dict()
        returned = sigpropy.TimeSeries.from_dict(dict_repr)
        self.assertEqual(expected.dt, returned.dt)
        self.assertArrayEqual(expected.amplitude, returned.amplitude)

    def test_to_and_from_json(self):
        dt = 1
        amplitude = np.array([1, 2, 3, 4])
        expected = sigpropy.TimeSeries(amplitude, dt)
        json_repr = expected.to_json()
        returned = sigpropy.TimeSeries.from_json(json_repr)
        self.assertEqual(expected.dt, returned.dt)
        self.assertArrayEqual(expected.amplitude, returned.amplitude)


if __name__ == '__main__':
    unittest.main()
