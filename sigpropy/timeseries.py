# This file is part of sigpropy, a Python package for signal processing.
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

"""TimeSeries class definition."""

import warnings
import logging
import json

import numpy as np
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt, detrend

logger = logging.getLogger("sigpropy.timeseries")

__all__ = ['TimeSeries']


class TimeSeries():
    """
    A class for manipulating time series.

    Attributes
    ----------
    amplitude : ndarray
        Denotes the time series amplitude one value per time step.
        Amplitude can be 1D or 2D, for the 2D case each row is a
        different time series.
    dt : float
        Time step between samples in seconds.

    """

    def __init__(self, amplitude, dt):
        """
        Initialize a `TimeSeries` object.

        Parameters
        ----------
        amplitude : ndarray
            Amplitude of the time series at each time step.
        dt : float
            Time step between samples in seconds.

        Returns
        -------
        TimeSeries
            Instantiated with amplitude information.

        Raises
        ------
        TypeError
            If `amplitude` is not castable to `ndarray` or has
            dimensions greater than 2. Refer to error message(s) for
            specific details.

        """
        # amplitude must be an ndarray of floating point numbers.
        try:
            self._amp = np.array(amplitude, dtype=np.double)
        except ValueError:
            msg = "`amplitude` must be convertable to numeric `ndarray`."
            raise TypeError(msg)

        # amplitude must be an ndarray with ndims=2.
        if self._amp.ndim == 1:
            self._amp = np.expand_dims(self._amp, axis=0)
        elif self._amp.ndim > 2:
            msg = f"`amplitude` must be 1-D or 2-D, not {self._amp.ndim}-D."
            raise TypeError(msg)
        else:
            pass

        self._dt = float(dt)

        logger.info("Creating a TimeSeries object.")
        logger.info(f"\tnsamples = {self.nsamples}")
        logger.info(f"\tdt = {self._dt}")

    @property
    def dt(self):
        return self._dt

    @property
    def amp(self):
        warnings.warn("`amp` is deprecated, use `amplitude` instead",
                      DeprecationWarning)
        return self._amp

    @property
    def n_samples(self):
        warnings.warn("`n_samples` is deprecated, use `nsamples` instead",
                      DeprecationWarning)
        return self.nsamples

    @property
    def nsamples(self):
        return self._amp.shape[1]

    @property
    def nsamples_per_window(self):
        return self._amp.shape[1]

    @property
    def windowlength(self):
        return (self.nsamples_per_window-1)*self.dt

    @property
    def n_windows(self):
        warnings.warn("`n_windows` is deprecated, use `nwindows` instead",
                      DeprecationWarning)
        return self.nwindows

    @property
    def nwindows(self):
        warnings.warn("`nwindows` is deprecated, use `nseries` instead",
                      DeprecationWarning)
        return self.nseries

    @property
    def nseries(self):
        return self._amp.shape[0]

    @property
    def fs(self):
        return 1/self._dt

    @property
    def fnyq(self):
        return 0.5*self.fs

    @property
    def df(self):
        return self.fs/self.nsamples

    @property
    def amplitude(self):
        return self._amp.squeeze()

    @property
    def time(self):
        start = 0
        delta = self.nsamples_per_window
        time = np.empty_like(self._amp)
        for row in range(self.nseries):
            time[row, :] = np.arange(start, start+delta)
            start += delta-1
        time *= self.dt
        return time.squeeze()

    def trim(self, start_time, end_time):
        """
        Trim time series in the interval [`start_time`, `end_time`].

        Parameters
        ----------
        start_time : float
            New time zero in seconds.
        end_time : float
            New end time in seconds.

        Returns
        -------
        None
            Updates the attributes `amplitude` and `nsamples`.

        Raises
        ------
        IndexError
            If the `start_time` and `end_time` is illogical.
            For example, `start_time` is before the start of the
            `delay` or after `end_time`, or the `end_time` is
            after the end of the record.

        """
        nseries_before_join = int(self.nseries)
        if self.nseries > 1:
            windowlength = self.windowlength
            warnings.warn("nseries > 1, so joining before splitting.")
            self.join()

        current_time = self.time
        start = min(current_time)
        end = max(current_time)

        if (start_time < start) or (start_time >= end_time):
            msg = f"Illogical start_time for trim. The following must be true: {start:.2f} < {start_time:.2f} < {end_time:.2f}."
            raise IndexError(msg)

        if (end_time > end):
            msg = f"Illogical end_time for trim. The following must be true: {start_time:.2f} < {end_time:.2f} < {end:.2f}."
            raise IndexError(msg)

        start_index = np.argmin(np.absolute(current_time - start_time))
        end_index = np.argmin(np.absolute(current_time - end_time))

        self._amp = self._amp[:, start_index:end_index+1]

        if nseries_before_join > 1:
            self.split(windowlength)

    def detrend(self):
        """
        Remove linear trend from time series.

        Returns
        -------
        None
            Removes linear trend from attribute `amplitude`.

        """
        self._amp = detrend(self._amp)

    def split(self, windowlength):
        """
        Split record into `n` series of length `windowlength`.

        Parameters
        ----------
        windowlength : float
            Duration of desired shorter series in seconds. If
            `windowlength` is not an integer multiple of `dt`, the
            window length is rounded to up to the next integer
            multiple of `dt`.

        Returns
        -------
        None
            Updates the object's internal attributes
            (e.g., `amplitude`).

        Notes
        -----
            The last sample of each window is repeated as the first
            sample of the following time window to ensure an intuitive
            number of windows. Without this, for example, a 10-minute
            record could not be broken into 10 1-minute records.

        Examples
        --------
            >>> import numpy as np
            >>> from sigpropy import TimeSeries
            >>> amp = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            >>> tseries = TimeSeries(amp, dt=1)
            >>> wseries = tseries.split(2)
            >>> wseries.amplitude
            array([[0, 1, 2],
                [2, 3, 4],
                [4, 5, 6],
                [6, 7, 8]])

        """
        if self.nseries > 1:
            warnings.warn("nseries > 1, so joining before splitting.")
            self.join()

        steps_per_win = int(windowlength/self.dt)
        nwindows = int((self.nsamples-1)/steps_per_win)
        rec_samples = (steps_per_win*nwindows)+1

        right_cols = np.reshape(self.amplitude[1:rec_samples],
                                (nwindows, steps_per_win))
        left_col = self.amplitude[:-steps_per_win:steps_per_win].T
        self._amp = np.column_stack((left_col, right_cols))

    def join(self):
        """
        Rejoin a split `TimeSeries`.

        Returns
        -------
        None
            Updates the object's internal attributes
            (e.g., `amplitude`).

        """
        nth = self.nsamples_per_window
        keep_ids = np.ones(self._amp.size, dtype=bool)
        keep_ids[nth::nth] = False
        self._amp = np.expand_dims(self._amp.flatten()[keep_ids], axis=0)

    def cosine_taper(self, width):
        """
        Apply cosine taper to time series.

        Parameters
        ----------
        width : {0.-1.}
            Amount of the time series to be tapered.
            `0` is equal to a rectangular and `1` a Hann window.

        Returns
        -------
        None
            Applies cosine taper to attribute `amplitude`.

        """
        self._amp = self._amp * tukey(self.nsamples, alpha=width)

    def bandpassfilter(self, flow, fhigh, order=5):
        """
        Apply bandpass Butterworth filter to time series.

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
            Filters attribute `amplitude`.

        """
        fnyq = self.fnyq
        b, a = butter(order, [flow/fnyq, fhigh/fnyq], btype='bandpass')
        self._amp = filtfilt(b, a, self._amp, padlen=3*(max(len(b), len(a))-1))

    @classmethod
    def from_trace(cls, trace):
        """
        Initialize a `TimeSeries` object from a trace object.

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

    @classmethod
    def from_dict(cls, dictionary):
        """
        Create `TimeSeries` object from dictionary representation.

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

    @classmethod
    def from_json(cls, json_str):
        """
        Instantiate `TimeSeries` object form Json string.

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

    @classmethod
    def from_timeseries(cls, timeseries):
        """
        Copy constructor for `TimeSeries` object.

        Parameters
        ----------
        timeseries : TimeSeries
            `TimeSeries` to be copied.

        Returns
        -------
        TimeSeries
            Copy of the provided `TimeSeries` object.

        """
        return cls(timeseries.amplitude, timeseries.dt)

    def to_json(self):
        """
        Json string representation of `TimeSeries` object.

        Returns
        -------
        str
            Json string with all of the relevant contents of the
            `TimeSeries`.

        """
        dictionary = self.to_dict()
        return json.dumps(dictionary)

    def to_dict(self):
        """
        Dictionary representation of `TimeSeries`.

        Returns
        -------
        dict
            Containing all of the relevant contents of the `TimeSeries`.

        """
        info = {}
        info["amplitude"] = [list(series) for series in self._amp]
        info["dt"] = self.dt

        return info

    def __eq__(self, other):
        """Check if `other` is equal to `self`."""
        for attr in ["nseries", "nsamples", "dt"]:
            if getattr(self, attr) != getattr(other, attr):
                return False

        my = self.amplitude
        ur = other.amplitude
        for my_val, ur_val in zip(my.flatten(), ur.flatten()):
            if my_val != ur_val:
                return False

        return True

    def __str__(self):
        """Human-readable representation of `TimeSeries`."""
        return f"TimeSeries of shape ({self.nseries},{self.nsamples}) at {id(self)}."

    def __repr__(self):
        """Unambiguous representation of `TimeSeries`."""
        return f"TimeSeries(amplitude={self.amplitude}, dt={self.dt})"
