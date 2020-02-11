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

"""This file contains a speed test to perform fft and smoothing."""

import sigpropy as sp
import obspy
import cProfile
import pstats
from testtools import get_full_path

full_path = get_full_path(__file__)

def main():
    fname = full_path+"data/a2/UT.STN11.A2_C50.miniseed"
    traces = obspy.read(fname)
    timeseries = sp.TimeSeries.from_trace(traces[0])
    timeseries.split(windowlength=120)
    timeseries.bandpassfilter(flow=0.2, fhigh=45, order=5)
    timeseries.cosine_taper(width=0.2)

    fft = sp.FourierTransform.from_timeseries(timeseries)
    fft.smooth_konno_ohmachi(bandwidth=40.)
    fft.resample(minf=0.1, maxf=50, nf=512, res_type="log", inplace=True)

fname = full_path+".tmp_profiler_run"
data = cProfile.run('main()', filename=fname)
stat = pstats.Stats(fname)
stat.sort_stats('tottime')
stat.print_stats(0.01)

# YEAR - MO - DY : TIME UNIT
# -------------------------
# 2020 - 02 - 10 :  1.532s -> Baseline
