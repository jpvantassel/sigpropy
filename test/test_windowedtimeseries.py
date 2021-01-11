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

    def test_cosine_taper(self):
        # 0% Window - (i.e., no taper)
        amp = np.ones(30)
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        wtseries = sigpropy.WindowedTimeSeries.from_timeseries(tseries, 9)
        wtseries.cosine_taper(0)
        expected = np.ones(10)
        for returned in wtseries.amp:
            self.assertArrayAlmostEqual(expected, returned, places=6)

        # 50% window
        amp = np.ones(30)
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        wtseries = sigpropy.WindowedTimeSeries.from_timeseries(tseries, 9)
        wtseries.cosine_taper(0.5)
        expected = np.array([0.000000000000000e+00, 4.131759111665348e-01,
                             9.698463103929542e-01, 1.000000000000000e+00,
                             1.000000000000000e+00, 1.000000000000000e+00,
                             1.000000000000000e+00, 9.698463103929542e-01,
                             4.131759111665348e-01, 0.000000000000000e+00])
        for returned in wtseries.amp:
            self.assertArrayAlmostEqual(expected, returned, places=6)

        # 100% Window
        amp = np.ones(30)
        dt = 1
        tseries = sigpropy.TimeSeries(amp, dt)
        wtseries = sigpropy.WindowedTimeSeries.from_timeseries(tseries, 9)
        wtseries.cosine_taper(1)
        expected = np.array([0.000000000000000e+00, 1.169777784405110e-01,
               4.131759111665348e-01, 7.499999999999999e-01,
               9.698463103929542e-01, 9.698463103929542e-01,
               7.500000000000002e-01, 4.131759111665350e-01,
               1.169777784405111e-01, 0.000000000000000e+00])
        for returned in wtseries.amp:
            self.assertArrayAlmostEqual(expected, returned, places=6)

    def test_from_trace(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trace = obspy.read(self.full_path+"data/vuws/1.dat")[0]
        tseries = sigpropy.TimeSeries.from_trace(trace)
        expected = sigpropy.WindowedTimeSeries.from_timeseries(tseries, 0.5)
        returned = sigpropy.WindowedTimeSeries.from_trace(trace, 0.5)
        self.assertEqual(expected, returned)

    def test_trim(self):

        def trim_test(start, end):
            amp = [[0,1,2,3],[3,4,5,6],[6,7,8,9]]
            dt = 1
            wtseries = sigpropy.WindowedTimeSeries(amp, dt)
            wtseries.trim(start, end)
            return wtseries.amp

        # No Trim
        _expected = np.array([[0,1,2,3],[3,4,5,6],[6,7,8,9]])
        expected = _expected
        returned = trim_test(0,9)
        self.assertArrayEqual(expected, returned)

        # First two windows
        expected = _expected[:2]
        returned = trim_test(0,6)
        self.assertArrayEqual(expected, returned)

        returned = trim_test(0,8)
        self.assertArrayEqual(expected, returned)

        # Only the Middle Window
        expected = _expected[1:2]
        returned = trim_test(2,8)
        self.assertArrayEqual(expected, returned)

        returned = trim_test(3,6)
        self.assertArrayEqual(expected, returned)

        # Last Window
        expected = _expected[2:3]
        returned = trim_test(5,10)
        self.assertArrayEqual(expected, returned)

        returned = trim_test(6,9)
        self.assertArrayEqual(expected, returned)

        # First Window
        expected = _expected[0:1]
        returned = trim_test(-1,4)
        self.assertArrayEqual(expected, returned)

        returned = trim_test(0,3)
        self.assertArrayEqual(expected, returned)

    def test_detrend(self):
        signal = np.array([0., .2, 0., -.2]*5)
        trend = np.arange(0, 20, 1)
        amp = signal + trend
        amp = np.vstack((amp, amp))
        dt = 1
        wtseries = sigpropy.WindowedTimeSeries(amp, dt)
        wtseries.detrend()
        expected = signal
        for returned in wtseries.amp:
            self.assertArrayAlmostEqual(expected, returned, delta=0.03)


if __name__ == '__main__':
    unittest.main()
