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

"""This file contains the class TimeSeries."""

import numpy as np
import json
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt, detrend
import logging
logger = logging.getLogger(__name__)


class TimeSeries():
    """A class for manipulating time series.

    Attributes
    ----------
    amp : ndarray
        Denotes the time series amplitude. If `amp` is 1D each 
        sample corresponds to a single time step. If `amp` is 2D 
        each row corresponds to a particular section of the time
        record (i.e., time window) and each column corresponds to a
        single time step.
    dt : float 
        Denotes the time step between samples in seconds.
    n_windows : int
        Number of time windows that the time series has been split
        into (i.e., number of rows of `amp` if 2D).
    n_samples : int
        Number of samples in time series (i.e., `len(amp)` if `amp`
        is 1D or number of columns if `amp` is 2D).
    fs : float
        Sampling frequency in Hz equal to `1/dt`.
    fnyq : float
        Nyquist frequency in Hz equal to `fs/2`. 
    """

    def __init__(self, amplitude, dt):
        """Intialize a `TimeSeries` object.

        Parameters
        ----------
        amplitude : ndarray 
            Amplitude of the time series at each time step. Refer to
            attribute definition for details.
        dt : float
            Time step between samples in seconds.

        Returns
        -------
        TimeSeries
            Instantiated with amplitude information.

        Raises
        ------
        ValueError
            If `delay` is greater than 0.
        """

        self.amp = TimeSeries._check_input("amplitude", amplitude)
        self.n_windows = 1 if len(self.amp.shape) == 1 else self.amp.shape[0]
        self.n_samples = len(self.amp)
        self.dt = dt
        self.fs = 1/self.dt
        self.fnyq = 0.5*self.fs
        self._df = self.fs/self.n_samples
        self._nstack = 1
        self.multiple = 1

        logger.info(f"Initialize a TimeSeries object.")
        logger.info(f"\tdt = {dt}")
        logger.info(f"\tfs = {self.fs}")
        logger.info(f"\tn_samples = {self.n_samples}")

    @staticmethod
    def _check_input(name, values):
        """Perform simple checks on values of parameter `name`.

        Specifically:
            1. Cast `values` to `ndarray`.
            2. Check that `ndarray` is no larger than 2D.

        Parameters
        ----------
        name : str
            `name` of parameter to be check. Only used to raise 
            easily understood exceptions.
        values : any
            value of parameter to be checked.

        Returns
        -------
        ndarray
            `values` cast to `ndarray`.

        Raises
        ------
        TypeError
            If entries do not comply with checks 1. and 2. listed above.

        """
        # Cast values to ndarray
        try:
            values = np.array(values, dtype=np.double)
        except ValueError:
            msg = f"{name} must be convertable to numeric `ndarray`."
            raise TypeError(msg)

        # Check dimension is <=2D
        if len(values.shape) > 2:
            msg = f"{name} must be 1D or 2D, not {values.shape}-dimensional."
            raise TypeError(msg)

        return values

    def to_dict(self):
        """Dictionary representation of `TimeSeries`.

        Returns
        -------
        dict
            Containing all of the relevant contents of the `TimeSeries`.
        """

        info = {}
        if self.n_windows > 1:
            msg = "This method is only implemented for `TimeSeries` with 1 window."
            raise NotImplementedError(msg)
        info["amplitude"] = list(self.amp)
        info["dt"] = self.dt

        return info

    @classmethod
    def from_dict(cls, dictionary):
        """Create `TimeSeries` object from dictionary representation.

        Parameters
        ----------
        dictionary : dict
            Must contain keys "amplitude" and "dt".

        Returns
        -------
        TimeSeries
            Instantiated `TimeSeries` object.

        Raises
        ------
        KeyError
            If any of the required keys (listed above) are missing.
        """

        return cls(dictionary["amplitude"], dictionary["dt"])

    def to_json(self):
        """Json string representation of `TimeSeries` object.

        Returns
        -------
        str
            Json string with all of the relevant contents of the 
            `TimeSeries`.
        """
        dictionary = self.to_dict()
        return json.dumps(dictionary)

    @classmethod
    def from_json(cls, json_str):
        """Instaniate `TimeSeries` object form Json string.

        Parameters
        ----------
        json_str : str
            Json string with all of the relevant contents of
            `TimeSeries`. Must contain keys "amplitude" and "dt".

        Returns
        -------
        TimeSeries
            Instantiated `TimeSeries` object.
        """

        dictionary = json.loads(json_str)
        return cls.from_dict(dictionary)

    @property
    def amplitude(self):
        return self.amp

    @property
    def time(self):
        """Return time vector for `TimeSeries` object."""
        if self.n_windows == 1:
            return np.arange(0, self.n_samples*self.dt, self.dt)
        else:
            samples_per_window = (self.n_samples//self.n_windows)+1
            time = np.zeros((self.n_windows, samples_per_window))
            for cwindow in range(self.n_windows):
                start_time = cwindow*(samples_per_window-1)*self.dt
                stop_time = start_time + (samples_per_window-1)*self.dt
                time[cwindow] = np.linspace(
                    start_time, stop_time, samples_per_window)
            return time

    def trim(self, start_time, end_time):
        """Trim excess from time series in the half-open interval
        [start_time, end_time).

        Parameters
        ----------
        start_time : float
            New time zero in seconds.
        end_time : float
            New end time in seconds. Note that the interval is
            half-open.

        Returns
        -------
        None
            Updates the attributes `n_samples`, `delay`, and `df`.

        Raises
        ------
        IndexError
            If the `start_time` and `end_time` is illogical.
            For example, `start_time` is before the start of the
            `delay` or after `end_time`, or the `end_time` is
            after the end of the record.
        """
        current_time = self.time
        start = min(current_time)
        end = max(current_time)

        if start_time < start or start_time > end_time:
            logger.debug(f"{start} < {start_time} < {end_time}: Must be True.")
            raise IndexError("Illogical start_time, see doctring.")

        if end_time > end or end_time < start_time:
            logger.debug(f"{start_time} < {end_time} < {end}: Must be True.")
            raise IndexError("Illogical end_time, see doctring.")

        logger.info(f"start = {start}, moving to start_time = {start_time}")
        logger.info(f"start = {end}, moving to end_time = {end_time}")

        start_index = np.argmin(np.absolute(current_time - start_time))
        end_index = np.argmin(np.absolute(current_time - end_time))

        logger.debug(f"start_index = {start_index}")
        logger.debug(f"start_index = {end_index}")

        self.amp = self.amp[start_index:end_index+1]
        self.n_samples = len(self.amp)
        self._df = self.fs/self.n_samples

        logger.info(f"n_samples = {self.n_samples}")
        logger.info(f"df = {self._df}")

    def detrend(self):
        """Remove linear trend from time series.

        Returns
        -------
        None
            Removes linear trend from attribute `amp`.
        """
        if self.n_windows == 1:
            self.amp = detrend(self.amp)
        else:
            for row, amp in enumerate(self.amp):
                self.amp[row] = detrend(amp)

    def split(self, windowlength):
        """Split time series into windows of duration `windowlength`.

        Parameters
        ----------
        windowlength : float
            Duration of desired window length in seconds. If 
            `windowlength` is not an integer multiple of `dt`, the 
            window length is rounded to up to the next integer
            multiple of `dt`.

        Returns
        -------
        None
            Reshapes attribute `amp` into a 2D array 
            where each row is a different consecutive time window and 
            each column denotes a time step. 

        Notes
        -----
            The last sample of each window is repeated as the first
            sample of the following time window to ensure a logical
            number of windows. Without this, a 10 minute record could
            not be broken into 10 1-minute records.

        Examples
        --------
            >>> import sigpropy as sp
            >>> import numpy as np
            >>> amp = np.array([0,1,2,3,4,5,6,7,8,9])
            >>> tseries = sp.TimeSeries(amp, dt=1) 
            >>> tseries.split(2)
            >>> tseries.amp
            array([[0, 1, 2],
                [2, 3, 4],
                [4, 5, 6],
                [6, 7, 8]])
        """
        new_points_per_win = int(windowlength/self.dt)
        self.n_windows = int((self.n_samples-1)/new_points_per_win)
        self.n_samples = (new_points_per_win*self.n_windows)+1
        tmp = np.reshape(self.amp[1:self.n_samples],
                         (self.n_windows, new_points_per_win))
        tmp_col = np.concatenate((np.array([self.amp[0]]), tmp[:-1, -1])).T
        self.amp = np.column_stack((tmp_col, tmp))

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
        if self.n_windows == 1:
            npts = self.n_samples
            self.amp = self.amp * tukey(npts, alpha=width)
        else:
            npts = self.amp.shape[1]
            cos_taper = tukey(npts, alpha=width)
            for c_window, c_amp in enumerate(self.amp):
                self.amp[c_window] = c_amp*cos_taper

    def bandpassfilter(self, flow, fhigh, order=5):
        """Apply bandpass Butterworth filter to time series.

        Parameters
        ----------
        flow : float
            Low-cut frequency (content below `flow` is filtered).
        fhigh : float
            High-cut frequency (content above `fhigh` is filtered).
        order : int, optional
            Filter order, default is 5.

        Returns
        -------
        None
            Filters attribute `amp`. 
        """
        b, a = butter(order, [flow/self.fnyq, fhigh/self.fnyq],
                      btype='bandpass')
        # TODO (jpv): Research padlen arguement
        self.amp = filtfilt(b, a, self.amp, padlen=3*(max(len(b), len(a))-1))

    @classmethod
    def from_trace(cls, trace, n_stacks=1, delay=0):
        """Initialize a `TimeSeries` object from a trace object.

        This method is more general method than `from_trace_seg2`, 
        as it does not attempt to extract any metadata from the `Trace` 
        object.

        Parameters
        ----------
        trace : Trace
            Refer to
            `obspy documentation <https://github.com/obspy/obspy/wiki>`_
            for more information

        Returns
        -------
        TimeSeries
            Initialized with information from `trace`.
        """
        return cls(amplitude=trace.data, dt=trace.stats.delta)

    def __eq__(self, other):
        my = self.amp
        ur = other.amp

        if my.size != ur.size:
            return False

        for my_val, ur_val in zip(my, ur):
            if my_val != ur_val:
                return False

        for attr in ["dt", "n_stacks", "delay"]:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __repr__(self):
        return f"TimeSeries(dt={self.dt}, amplitude={str(self.amp[0:3])[:-1]} ... {str(self.amp[-3:])[1:]})"
