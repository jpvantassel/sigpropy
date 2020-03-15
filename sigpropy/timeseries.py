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
        Denotes the time series amplitude one value per time step.
    dt : float 
        Denotes the time step between samples in seconds.
    """

    def __init__(self, amplitude, dt):
        """Intialize a `TimeSeries` object.

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
            dimenension not equal to 1. See error message(s) for
            details.
        """
        print("TimeSeries __init__")
        self.amp = TimeSeries._check_input("amplitude", amplitude)
        self._dt = dt

        logger.info(f"Initialize a TimeSeries object.")
        logger.info(f"\tnsamples = {self.nsamples}")
        logger.info(f"\tdt = {self._dt}")

    @property
    def dt(self):
        return self._dt

    @property
    def n_samples(self):
        return self.nsamples

    @property
    def nsamples(self):
        return len(self.amp)

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
        return self.amp

    @property
    def time(self):
        return np.arange(0, self.nsamples*self.dt, self.dt)

    @staticmethod
    def _check_input(name, values):
        """Perform simple checks on values of parameter `name`.

        Specifically:
            1. Cast `values` to `ndarray`.
            2. Check that `ndarray` is 1D.

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

        if len(values.shape) != 1:
            msg = f"{name} must be 1-D, not {len(values.shape)}-D."
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

    def detrend(self):
        """Remove linear trend from time series.

        Returns
        -------
        None
            Removes linear trend from attribute `amp`.
        """
        self.amp = detrend(self.amp)

    def split(self, windowlength):
        msg = "This method has been removed refer to class WindowTimeSeries"
        raise DeprecationWarning(msg)

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
        self.amp = self.amp * tukey(self.nsamples, alpha=width)

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
        #TODO 
        fnyq = self.fnyq
        b, a = butter(order, [flow/fnyq, fhigh/fnyq], btype='bandpass')
        # TODO (jpv): Research padlen arguement
        self.amp = filtfilt(b, a, self.amp, padlen=3*(max(len(b), len(a))-1))

    @classmethod
    def from_trace(cls, trace):
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
        print("From trace in timeseries")
        return cls(amplitude=trace.data, dt=trace.stats.delta)

    def __eq__(self, other):
        my = self.amp
        ur = other.amp

        if my.size != ur.size:
            return False

        for my_val, ur_val in zip(my.flatten(), ur.flatten()):
            if my_val != ur_val:
                return False

        for attr in ["dt"]:
            if getattr(self, attr) != getattr(other, attr):
                return False
        return True

    def __repr__(self):
        return f"TimeSeries(dt={self.dt}, amplitude={str(self.amp[0:3])[:-1]} ... {str(self.amp[-3:])[1:]})"
