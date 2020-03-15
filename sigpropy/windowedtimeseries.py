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

"""This file contains the class WindowedTimeSeries."""

import numpy as np
from sigpropy import TimeSeries
from scipy.signal.windows import tukey
from scipy.signal import detrend
import logging
logger = logging.getLogger(__name__)


class WindowedTimeSeries(TimeSeries):
    """A class for time series that has been split into windows

    Attributes
    ----------
    amp : ndarray
        Time series amplitude. Each row corresponds to a particular
        section (i.e., window) of the time series and each column
        corresponds to a time step.
    dt : float 
        Denotes the time step between samples in seconds.

    nwindows : int
        Number of time windows that the time series has been split
        into (i.e., number of rows of `amp`).
    """

    def __init__(self, amplitude, dt):
        """Initialize a `WindowedTimeSeries` object.

        Parameters
        ----------
        amplitude : ndarray
            2D array of amplitudes. Each row corresponds to a different
            time window and each column a time step. Note that the last
            and first samples of consecutive windows are shared.
        dt : float
            Time step between samples in seconds.

        Returns
        -------
        WindowedTimeSeries
            Instantiated with amplitude information.

        Raises
        ------
        ValueError
            If `delay` is greater than 0.
        """
        self.amp = WindowedTimeSeries._check_input("amplitude", amplitude)
        self._dt = dt

        logger.info(f"Initialize a WindowedTimeSeries object.")
        logger.info(f"\tshape = {self.amp.shape}")
        logger.info(f"\tdt = {self._dt}")

    @staticmethod
    def _check_input(name, values):
        """Perform simple checks on values of parameter `name`.

        Specifically:
            1. Cast `values` to `ndarray`.
            2. Check that `ndarray` is 2D.

        Parameters
        ----------
        name : str
            Name of parameter to be check. Only used to raise 
            easily understood exceptions.
        values : any
            Value of parameter to be checked.

        Returns
        -------
        ndarray
            `values` cast to `ndarray`.

        Raises
        ------
        TypeError
            If entries do not comply with checks 1. and 2. listed above.
        """
        try:
            values = np.array(values, dtype=np.double)
        except ValueError:
            msg = f"{name} must be convertable to numeric `ndarray`."
            raise TypeError(msg)

        if len(values.shape) != 2:
            msg = f"{name} must be 2-D, not {len(values.shape)}-D."
            raise TypeError(msg)

        return values

    @property
    def time(self):
        nsamples_per_window = self.nsamples_per_window
        windowlength = self.windowlength
        time = np.empty_like(self.amp)
        # TODO (jpv): Try using enumerate here.
        for window_index in range(self.nwindows):
            start = window_index*windowlength
            stop = start + windowlength
            time[window_index] = np.linspace(start, stop, nsamples_per_window)
        return time

    @property
    def nsamples(self):
        return len(self.amp)

    @property
    def n_windows(self):
        return self.nwindows

    @property
    def nwindows(self):
        return self.amp.shape[0]

    @property
    def nsamples_per_window(self):
        return self.amp.shape[1]

    @property
    def windowlength(self):
        return (self.nsamples_per_window-1)*self.dt

    @classmethod
    def from_timeseries(cls, timeseries, windowlength):
        """Create `WindowedTimeSeries` from `TimeSeries` object.

        Parameters
        ----------
        timeseries : TimeSeries
            Time series to be divided into time windows.
        windowlength : float
            Duration of desired windows in seconds. If 
            `windowlength` is not an integer multiple of `dt`, the 
            window length is rounded to up to the next integer
            multiple of `dt`.

        Returns
        -------
        WindowedTimeSeries
            Initialized `WindowedTimeSeries` object.

        Notes
        -----
            The last sample of each window is repeated as the first
            sample of the following time window to ensure an inuitive
            number of windows. Without this, a 10 minute record could
            not be broken into 10 1-minute records.

        Examples
        --------
            >>> from sigpropy import TimeSeries, WindowedTimeSeries
            >>> import numpy as np
            >>> amp = np.array([0,1,2,3,4,5,6,7,8,9])
            >>> tseries = TimeSeries(amp, dt=1) 
            >>> wseries = WindowedTimeSeries.from_timeseries(tseries, 2)
            >>> wseries.amp
            array([[0, 1, 2],
                [2, 3, 4],
                [4, 5, 6],
                [6, 7, 8]])
        """
        steps_per_win = int(windowlength/timeseries.dt)
        nwindows = int((timeseries.nsamples-1)/steps_per_win)
        rec_samples = (steps_per_win*nwindows)+1

        right_cols = np.reshape(timeseries.amp[1:rec_samples],
                                (nwindows, steps_per_win))
        left_col = timeseries.amp[:-steps_per_win:steps_per_win].T
        amp = np.column_stack((left_col, right_cols))

        return cls(amp, timeseries.dt)

    def to_dict(self):
        msg = "This method has not been implemented."
        raise NotImplementedError(msg)

    def trim(self, start_time, end_time):
        """Trim WindowedTimeSeries.

        Parameters
        ----------
        start_time, end_time : float
            Desired start and end times of the new WindowedTimeSeries in
            seconds. The `start_time` should correpsond to the first
            sample of the timewindow otherwise the entire window will be
            rejected (i.e., no partial windows are allowed), likewise
            the `end_time` should correspond to the last time
            of the window otherwise the entire window will be rejected.

        Returns
        -------
        None
            Modifies the attribute `amp`.

        Raises
        ------
        ValueError
            If `start_time` and `end_time` are illogical.
        """
        time_array = self.time
        delta = self.dt/2
        effective_start_time = start_time - delta
        effective_end_time = end_time + delta

        for start_index, time_window in enumerate(time_array):
            if time_window[0] > effective_start_time:
                break

        for end_index, time_window in enumerate(time_array):
            if time_window[-1] > effective_end_time:
                break
        else:
            end_index += 1
        
        if start_index > end_index:
            msg = f"end_time ({end_time}) must be greater than start_time ({start_time})"
            raise ValueError(msg)

        self.amp = self.amp[start_index:end_index]

    def detrend(self):
        """Remove linear trend from time series.

        Returns
        -------
        None
            Removes linear trend from attribute `amp`.
        """
        for row, amp in enumerate(self.amp):
            self.amp[row] = detrend(amp)

    def cosine_taper(self, width):
        """Apply cosine taper to time series.

        Parameters
        ----------
        width : {0.-1.}
            Amount of the time series to be tapered.
            `0` is equal to a rectangular and `1` a Hann window.

        Returns
        -------
        None
            Applies cosine taper to attribute `amp`.
        """
        cos_taper = tukey(self.nsamples_per_window, alpha=width)
        for c_window, c_amp in enumerate(self.amp):
            self.amp[c_window] = c_amp*cos_taper

    @classmethod
    def from_trace(cls, trace, windowlength):
        """Initialize a `WindowedTimeSeries` object from a trace object.

        Parameters
        ----------
        trace : Trace
            Refer to
            `obspy documentation <https://github.com/obspy/obspy/wiki>`_
            for more information

        Returns
        -------
        WindowedTimeSeries
            Initialized with information from `trace` and split into
            windows of length `windowlength`.
        """
        timeseries = TimeSeries.from_trace(trace)
        return cls.from_timeseries(timeseries, windowlength)

    def __eq__(self, other):
        if not TimeSeries.__eq__(self, other):
            return False

        for attr in ["nwindows"]:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True
