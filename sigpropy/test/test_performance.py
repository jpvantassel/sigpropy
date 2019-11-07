"""This file contains a speed test for the main hvsr calculation."""

import sigpropy as sp
import obspy
import cProfile
import pstats

def main():
    fname = "test/data/a2/UT.STN11.A2_C50.miniseed"
    traces = obspy.read(fname)
    timeseries = sp.TimeSeries.from_trace(traces[0])
    timeseries.split(windowlength=120)
    timeseries.bandpassfilter(flow=0.2, fhigh=45, order=5)
    timeseries.cosine_taper(width=0.2)

    fft = sp.FourierTransform.from_timeseries(timeseries)
    # fft.resample(minf=0.01, maxf=max(fft.frq), nf=3000, res_type="linear", inplace=True)
    fft.smooth_konno_ohmachi(bandwidth=40.)
    fft.resample(minf=0.1, maxf=50, nf=512, res_type="log", inplace=True)

fname = "test/.tmp_profiler_run"
data = cProfile.run('main()', filename=fname)
stat = pstats.Stats(fname)
stat.sort_stats('tottime')
stat.print_stats(0.01)