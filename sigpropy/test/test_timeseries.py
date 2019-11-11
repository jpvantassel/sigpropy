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

"""Tests for TimeSeries class. """

import matplotlib.pyplot as plt
import unittest
import sigpropy
import obspy
import numpy as np
import warnings
import logging
logging.basicConfig(level=logging.WARNING)


class TestTimeSeries(unittest.TestCase):

    def test_check(self):
        for value in [True, "values", 1, 1.57]:
            self.assertRaises(TypeError,
                              sigpropy.TimeSeries._check_input,
                              name="blah",
                              values=value)
        for value in [[1, 2, 3], (1, 2, 3)]:
            value = sigpropy.TimeSeries._check_input(name="blah", values=value)
            self.assertTrue(isinstance(value, np.ndarray))

        # for value in [[[1, 2], [3, 4]], ((1, 2), (3, 4)), np.array([[1, 2], [3, 4]])]:
        #     self.assertRaises(TypeError,
        #                       sigpropy.TimeSeries._check_input,
        #                       name="blah",
        #                       values=value)

    def test_init(self):
        dt = 1
        amp = [0, 1, 0, -1]
        test = sigpropy.TimeSeries(amp, dt)
        self.assertListEqual(amp, test.amp.tolist())
        self.assertEqual(dt, test.dt)

        amp = np.array(amp)
        test = sigpropy.TimeSeries(amp, dt)
        self.assertListEqual(amp.tolist(), test.amp.tolist())

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

    def test_split(self):
        amp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dt = 1
        thist = sigpropy.TimeSeries(amp, dt)
        thist.split(2)
        self.assertListEqual(np.array([[1, 2, 3],
                                       [3, 4, 5],
                                       [5, 6, 7],
                                       [7, 8, 9]]).tolist(),
                             thist.amp.tolist())

        amp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dt = 0.5
        thist = sigpropy.TimeSeries(amp, dt)
        thist.split(1)
        self.assertListEqual(np.array([[1, 2, 3],
                                       [3, 4, 5],
                                       [5, 6, 7],
                                       [7, 8, 9]]).tolist(),
                             thist.amp.tolist())

        amp = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        dt = 1
        thist = sigpropy.TimeSeries(amp, dt)
        thist.split(4)
        self.assertListEqual(np.array([[1, 2, 3, 4, 5],
                                       [5, 6, 7, 8, 9]]).tolist(),
                             thist.amp.tolist())

    def test_cosine_taper(self):
        # 0% Window - (i.e., no taper)
        amp = np.ones(10)
        dt = 1
        thist = sigpropy.TimeSeries(amp, dt)
        thist.cosine_taper(0)
        sol = amp
        for test, known in zip(thist.amp, sol):
            self.assertAlmostEqual(test, known, places=6)

        # 50% window
        amp = np.ones(10)
        dt = 1
        thist = sigpropy.TimeSeries(amp, dt)
        thist.cosine_taper(0.5)
        sol = [0.000000000000000e+00, 4.131759111665348e-01,
               9.698463103929542e-01, 1.000000000000000e+00,
               1.000000000000000e+00, 1.000000000000000e+00,
               1.000000000000000e+00, 9.698463103929542e-01,
               4.131759111665348e-01, 0.000000000000000e+00]
        for test, known in zip(thist.amp, sol):
            self.assertAlmostEqual(test, known, places=6)

        # 100% Window
        amp = np.ones(10)
        dt = 1
        thist = sigpropy.TimeSeries(amp, dt)
        thist.cosine_taper(1)
        sol = [0.000000000000000e+00, 1.169777784405110e-01,
               4.131759111665348e-01, 7.499999999999999e-01,
               9.698463103929542e-01, 9.698463103929542e-01,
               7.500000000000002e-01, 4.131759111665350e-01,
               1.169777784405111e-01, 0.000000000000000e+00]
        for test, known in zip(thist.amp, sol):
            self.assertAlmostEqual(test, known, places=6)

    def test_from_trace(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = obspy.read("test/data/vuws/1.dat")[0]
        tseries = sigpropy.TimeSeries.from_trace(trace, delay=-0.5)
        self.assertListEqual(tseries.amp.tolist(), trace.data.tolist())
        self.assertEqual(tseries.dt, trace.stats.delta)
        self.assertEqual(tseries._nstack, int(trace.stats.seg2.STACK))
        self.assertEqual(tseries.delay, float(trace.stats.seg2.DELAY))

    def test_zero_pad(self):
        thist = sigpropy.TimeSeries(amplitude=list(np.arange(0, 2, 0.01)),
                                    dt=0.01)
        self.assertEqual(len(thist.amp), 200)
        thist.zero_pad(df=0.1)
        self.assertEqual(len(thist.amp), 1000)
        thist.zero_pad(df=0.5)
        self.assertEqual(len(thist.amp)/thist.multiple, 1/(0.01*0.5))

        thist = sigpropy.TimeSeries(amplitude=list(np.arange(0, 2, 0.02)),
                                    dt=0.02)
        self.assertEqual(len(thist.amp), 100)
        thist.zero_pad(df=1.)
        self.assertEqual(len(thist.amp), 200)
        self.assertEqual(thist.multiple, 4)
        self.assertEqual(len(thist.amp)/thist.multiple, 1/(0.02*1))

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
        signal =np.array([0., .2, 0., -.2]*5)
        trend = np.arange(0,20,1)
        amp = signal + trend
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        tseries.detrend()
        for true, test in zip(signal, tseries.amp):
            self.assertAlmostEqual(true, test, delta=0.03)

        # 2d amp
        signal = np.array([0., .2, 0., -.2]*5)
        trend = np.arange(0,20,1)
        amp = signal + trend
        amp = np.vstack((amp, amp))
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        tseries.detrend()
        for row in tseries.amp:
            print(row)
            for true, test in zip(signal, row):
                self.assertAlmostEqual(true, test, delta=0.03)


if __name__ == '__main__':
    unittest.main()
