"""Tests for FourierTransform class."""

import unittest
import sigpropy
import numpy as np
import matplotlib.pyplot as plt


class TestFourierTransform(unittest.TestCase):
    def test_init(self):
        dt = 0.005
        amp = np.sin(2*np.pi*5*np.arange(1000)*dt)
        mythist = sigpropy.TimeSeries(amp, dt)
        # plt.plot(mythist.time, mythist.amp)
        # plt.show()
        myfft = sigpropy.FourierTransform.from_timeseries(mythist)
        # plt.plot(myfft.frq, myfft.mag)
        myfft.smooth_konno_ohmachi()
        # plt.plot(myfft.frq, myfft.mag)
        # plt.xscale('log')
        # plt.show()


if __name__ == "__main__":
    unittest.main()
