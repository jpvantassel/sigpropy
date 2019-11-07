"""This file contains the class TimeSeries for creating and manipulating
time series objects.
"""

import numpy as np
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt
import logging
logger = logging.getLogger(__name__)


class TimeSeries():
    """A class for editing and manipulating time series.

    Attributes:
        amp: np.array
            Denotes the time series amplitude.
        dt: float 
            Denotes the time step between samples in seconds.
    """
    @staticmethod
    def check_input(name, values):
        """Perform simple checks on values of parameter `name`.

        Specifically:
            1. Check `values` is `np.ndarray`, `list`, or `tuple`. 
            2. If `list` or `tuple` convert to a `np.ndarray`.
            3. Check `values` is one-dimensional `np.ndarray`.

        Args:
            name : str
                `name` of parameter to be check. Only used to raise 
                easily understood exceptions.
            values : any
                value of parameter to be checked.

        Returns:
            `values` which may be transformed to an `np.array`.
        """
        if type(values) not in [np.ndarray, list, tuple]:
            raise TypeError(
                f"{name} must be of type np.array, list, or tuple not {type(values)}.")

        if isinstance(values, (list, tuple)):
            values = np.array(values)

        if len(values.shape) > 1:
            raise TypeError(
                f"{name} must be 1-dimensional, not {values.shape}-dimensional.")
        return values

    def __init__(self, amplitude, dt, nstacks=1, delay=0):
        """Initialize a TimeSeries object.

        Args:
            amplitude : np.1darray 
                Amplitude of the timeseries with time. The amplitude[0]
                is associated with first and amplitude[-1] is 
                associated with the final time sample.
            dt: float
                Time step between samples in seconds.
            nstacks: int, optional
                Number of stacks used to produce the amplitude (the
                default value is 1, denoting a single recording).
            delay: float
                Indicates the pre-event delay in seconds.

        Returns:
            Intialized TimeSeries object.
        """
        self.amp = TimeSeries.check_input("amplitude", amplitude)
        self.n_windows = 1
        self.n_samples = len(self.amp)
        self.dt = dt
        self.fs = 1/self.dt
        self.fnyq = 0.5*self.fs
        self.df = self.fs/self.n_samples
        self._nstack = 1
        assert(delay <= 0)
        self.delay = delay
        self.multiple = 1

        logging.info(f"Initialize a TimeSeries object.")
        logging.info(f"\tdt = {dt}")
        logging.info(f"\tfs = {self.fs}")
        logging.info(f"\tdelay = {delay}")
        logging.info(f"\tn_samples = {self.n_samples}")

    @property
    def time(self):
        """Return time vector for TimeSeries object."""
        return np.arange(0, self.n_samples*self.dt, self.dt) + self.delay

    def trim(self, start_time, end_time):
        """Trim excess from TimeSeries object in the half-open interval
        [start_time, end_time).

        Args: 
            start_time : float
                New time zero in seconds.
            end_time : float
                New end time in seconds. Note that the interval is
                half-open.

        Returns:
            Returns `None`, but may update the attributes: `n_samples`, 
            `delay`, and `df`.

        Raises:
            IndexError If the start_time and end_time is illogical.
                For example, start_time is before the start of the
                pretrigger delay or after end_time, or the end_time is
                before the start_time or after the end of the record.
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

        self.amp = self.amp[start_index:end_index]
        self.n_samples = len(self.amp)
        self.df = self.fs/self.n_samples
        self.delay = 0 if start_time >= 0 else start_time

        logger.info(f"n_samples = {self.n_samples}")
        logger.info(f"df = {self.df}")
        logger.info(f"delay = {self.delay}")

    def zero_pad(self, df=0.2):
        """Append zeros to the end of the TimeSeries object to achieve a
        desired frequency step.

        Args:
            df : float
                Frequency step in Hertz must be positive.

        Returns:
            Returns `None`, instead modifies attributes: `amp`,
            `n_samples`, and `multiple`.

        Raises:
            TypeError
                If `df` is not a float.
            ValueError
                If `df` is not positive.
        """
        if isinstance(df, float):
            if df <= 0:
                raise ValueError(f"`df` must be positive.")
        else:
            raise TypeError(f"`df` must be `float`, not {type(df)}.")

        nreq = int(np.round(1/(df*self.dt)))

        logging.info(f"nreq = {nreq}")

        # If nreq > n_samples, padd zeros to achieve n_samples = nreq
        if nreq > self.n_samples:
            self.amp = np.concatenate((self.amp,
                                       np.zeros(nreq - self.n_samples)))
            self.n_samples = nreq

        # If nreq <= n_samples, padd zeros to achieve a fraction of df (e.g., df/2)
        # After processing, extract the results at the frequencies of interest
        else:
            # Multiples of df and nreq
            multiples = np.array([1, 2, 4, 8, 16, 32], int)
            trial_n = nreq*multiples

            logging.debug(f"trial_n = {trial_n}")

            # Find first trial_n > nreq
            self.multiple = multiples[np.where(self.n_samples < trial_n)[0][0]]
            logging.debug(f"multiple = {self.multiple}")

            self.amp = np.concatenate((self.amp,
                                       np.zeros(nreq*self.multiple - self.n_samples)))
            self.n_samples = nreq*self.multiple

            logging.debug(f"n_samples = {self.n_samples}")

    def split(self, windowlength):
        """Split TimeSeries into windows of length windowlength.

        Args:
            windowlength : int
                Integer defining length of window in seconds. If 
                `windowlength` is not integer multiple of dt, the 
                window length is rounded to up to the next integer
                multiple of dt.

        Returns:
            Returns `None`, reshapes attribute `amp` into a 2D array 
            where each row is a different consecutive time window and 
            each column denotes a time step. 

        Note:
            The last sample of each window is repeated as the first
            sample of the following time window to ensure a logical
            number of windows. Without this a 10 minute record could
            not be broken into 10 1-minute records.

        Example:
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
        """Apply cosine taper to TimeSeries.

        Args:
            width : float {0.-1.} 
                Amount of the TimeSeries to be tapered. 0 represents a 
                rectangular window and 1 a Hann window.

        Returns:
            Returns `None`, instead applies cosine taper to attribute
            `amp`.
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
        """Bandpass filter TimeSeries.

        Args:
            flow: float
                Low cut-off frequency for filter (content below flow is 
                filtered).
            fhigh: float
                High cut-off frequency for filter (content above fhigh is
                filtered).
            order: int
                Filter order

        Returns:
            Returns `None`, perform filter operation on attribute `amp`. 
        """
        b, a = butter(order, [flow/self.fnyq, fhigh/self.fnyq], btype='bandpass')
        # TODO (jpv): Research padlen arguement
        self.amp = filtfilt(b, a, self.amp, padlen=3*(max(len(b), len(a))-1))

    @classmethod
    def from_trace(cls, trace, nstacks=1, delay=0):
        """Initialize a TimeSeries object from a trace object.

        This method is a more general method than from_trace_seg2, which
        does not attempt to extract anything from the trace object 
        except for its data and sampling step.

        Args:
            trace: Trace object.

            nstacks: Integer representing the number of stacks. Default 
                value is one (i.e., only a single recording was made).

            delay: Number that is less than or equal to zero, denoting
                the pre-trigger delay. Default value is zero (i.e., no
                pre-trigger delay was recorded).

        Returns:
            Initialized TimeSeries object.

        Raises:
            This method raises no exceptions.
        """
        return cls(amplitude=trace.data,
                   dt=trace.stats.delta,
                   nstacks=nstacks,
                   delay=delay)

    def __repr__(self):
        """Valid python expression to reproduce the object"""
        return f"TimeSeries(dt={self.dt}, amplitude={str(self.amp[0:3])[:-1]} ... {str(self.amp[-3:])[1:]})"
