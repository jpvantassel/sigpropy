"""This file contains the class FourierTransform for creating and 
working with fourier transform objects."""

import numpy as np
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
import scipy.interpolate as sp
# from numba import jit

def center_zero(frequencies):
    smoothing_window = np.zeros(len(frequencies), dtype=frequencies.dtype)
    smoothing_window[frequencies == 0.0] = 1.0
    return smoothing_window

# @jit(nopython=True)
def parta(frequencies, center_frequency, bandwidth):
    return bandwidth * np.log10(frequencies / center_frequency)

# @jit(nopython=True)
def partb(frequencies, center_frequency, bandwidth, smoothing_window):
    return np.sin(smoothing_window) / smoothing_window

# @jit(nopython=True)
def partc(frequencies, center_frequency, bandwidth, smoothing_window):
    return smoothing_window * smoothing_window * smoothing_window * smoothing_window

def makewindow(frequencies, center_frequency, bandwidth=40.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        smoothing_window = parta(frequencies, center_frequency, bandwidth)
        smoothing_window = partb(frequencies, center_frequency, bandwidth, smoothing_window)
        smoothing_window = partc(frequencies, center_frequency, bandwidth, smoothing_window)
    return smoothing_window

def fix_window(frequencies, center_frequency, smoothing_window):
    smoothing_window[frequencies == center_frequency] = 1.0
    smoothing_window[frequencies == 0.0] = 0.0
    return smoothing_window

class FourierTransform():
    """A class for editing and manipulating fourier transforms.

    Attributes:
        frq : np.array
            Frequency vector of the transform in Hertz.

        amp : np.array
            The transform's amplitude in the same units as the input.
    """

    @staticmethod
    def fft(amplitude, dt):
        """Compute the fast-fourier transform of a time series.

        Args:
            amplitude : np.array
                Time series amplitudes (one per time step). Can be a 2D
                array where each row is a valid time series.

            dt : float
                Indicating the time step in seconds.

        Returns:
            Tuple of the form (frq, fft) where:
            frq is an np.array containing the positve frequency vector
                between zero and the nyquist frequency (if even) or near
                the nyquist (if odd) in Hertz.
            fft is an np.array of complex amplitudes for the frequencies
                between zero and the nyquist with units of the input 
                ampltiude. If `amplitude` is a 2D array `fft` will also 
                be a 2D array where each row is the fft of each row of
                `amplitude`.
        """
        if len(amplitude.shape) > 2:
            raise TypeError("`amplitude` cannot have dimension > 2.")

        npts = amplitude.shape[-1]
        nfrqs = int(npts/2)+1 if (npts % 2) != 0 else int(npts+1/2)
        frq = np.abs(np.fft.fftfreq(npts, dt))[0:nfrqs]
        if len(amplitude.shape) == 1:
            return(2/npts * np.fft.fft(amplitude)[0:nfrqs], frq)
        else:
            fft = np.zeros((amplitude.shape[0], nfrqs), dtype=complex)
            for cwindow, amplitude in enumerate(amplitude):
                fft[cwindow] = 2/npts * np.fft.fft(amplitude)[0:nfrqs]
            return (fft, frq)

    def __init__(self, amplitude, frq, fnyq=None):
        """Initialize a FourierTransform object.

        Args:
            amplitude: np.array
                Fourier transform amplitude.
            frq: np.array 
                Linearly spaced frequency vector for fourier transform.
            fnyq: float, optional
                Nyquist frequency of Fourier Transform (by default the
                maximum value of frq vector is used)

        Returns:
            An initialized FourierTransform object.
        """
        self.amp = amplitude
        self.frq = frq
        self.fnyq = fnyq if fnyq != None else np.max(self.frq)

    def smooth_konno_ohmachi(self, bandwidth=40.0):
        # self.amp = konno_ohmachi_smoothing(self.mag, self.frq, bandwidth,
        #                                    enforce_no_matrix=False, max_memory_usage=2048)
        self.amp = self.mag
        smooth_amp = np.zeros(self.amp.shape)
        if len(self.amp.shape) == 1:
            for cid, cfrq in enumerate(self.frq):
                smoothing_window = self._k_and_o_window(self.frq, cfrq,
                                                        bandwidth=bandwidth)
                smooth_amp[cid] = np.dot(self.amp, smoothing_window)
        else:
            for c_col, cfrq in enumerate(self.frq):
                smoothing_window = self._k_and_o_window(self.frq, cfrq,
                                                        bandwidth=bandwidth)
                for c_row, c_amp in enumerate(self.amp):
                    smooth_amp[c_row, c_col] = np.dot(c_amp, smoothing_window)
        self.amp = smooth_amp

    @staticmethod
    def _k_and_o_window(frequencies, center_frequency,
                        bandwidth=40.0, normalize=False):
        if frequencies.dtype != np.float32 and frequencies.dtype != np.float64:
            msg = 'frequencies needs to have a dtype of float32/64.'
            raise ValueError(msg)

        if center_frequency == 0:
            return center_zero(frequencies) 

        smoothing_window = makewindow(frequencies, center_frequency)

        smoothing_window = fix_window(frequencies, center_frequency, smoothing_window)

        if normalize:
            smoothing_window /= smoothing_window.sum()

        return smoothing_window

    # @staticmethod
    # def _k_and_o_window(frequencies, center_frequency, bandwidth=40.0):
    #     # if frequencies.dtype != np.float32 and frequencies.dtype != np.float64:
    #     #     msg = 'frequencies needs to have a dtype of float32/64.'
    #     #     raise ValueError(msg)
    #     # If the center_frequency is 0 return an array with zero everywhere except
    #     # at zero.

    #     if center_frequency == 0:
    #         # smoothing_window = np.zeros(len(frequencies), dtype=frequencies.dtype)
    #         # smoothing_window[frequencies == 0.0] = 1.0
    #         return center_zero(frequencies, center_frequency, bandwidth)

    #     # Disable div by zero errors and return zero instead
    #     # with np.errstate(divide='ignore', invalid='ignore'):
    #     #     # Calculate the bandwidth*log10(f/f_c)
    #     #     # smoothing_window = bandwidth * np.log10(frequencies / center_frequency)
    #     #     # # Just the Konno-Ohmachi formulae.
    #     #     # smoothing_window[:] = (np.sin(smoothing_window) / smoothing_window) ** 4

    #     #     # Calculate the bandwidth*log10(f/f_c)
    #     #     smoothing_window = bandwidth * np.log10(frequencies / center_frequency)
    #     #     # # Just the Konno-Ohmachi formulae.
    #     #     smoothing_window = (np.sin(smoothing_window) / smoothing_window) ** 4

    #     smoothing_window = makewindow(frequencies, center_frequency, bandwidth)

    #     smoothing_window = fix_window(frequencies, center_frequency, smoothing_window)
    #     # Normalize to one if wished.
    #     # if normalize:
    #     #     smoothing_window /= smoothing_window.sum()
    #     return smoothing_window

    def resample(self, minf, maxf, nf, res_type="log", inplace=False):
        """Resample FourierTransform over a specified range.

        Args:
            minf : float 
                Minimum value of resample.
            maxf : float
                Maximum value or resample.
            nf : int
                Number of resamples.
            res_type : {"log", "linear"}
                Type of resampling.
            inplace : bool
                Determines whether resampling is done in place or 
                if a copy should be returned.

        Returns:
            If inplace=True, None.

            If inplace=False, a tuple of the form (frequency, ) where each parameter is a list.

        Raises:
            ValueError
                If `maxf`, `minf`, or `nf` are illogical.
            NotImplementedError
                If `res_type` is not amoung those options specified.
        """
        if maxf < minf:
            raise ValueError("`maxf` must be > `minf`")
        if type(nf) not in (int,):
            raise TypeError("`nf` must be postive integer")
        if nf < 0:
            raise TypeError("`nf` must be postive integer")
        types = {"log": "log", "linear": "linear"}
        if maxf > self.fnyq:
            raise ValueError("`maxf` is out of range.")
        if minf < 0:
            raise ValueError("`minf` is out of range.")

        if types[res_type] == types["log"]:
            xx = np.logspace(np.log10(minf), np.log10(maxf), nf)
        elif types[res_type] == types["linear"]:
            xx = np.linspace(minf, maxf, nf)
        else:
            raise NotImplementedError(
                f"{res_type} resampling has not been implemented.")

        interp_amp = sp.interp1d(self.frq, self.amp, kind="linear",
                                 fill_value="extrapolate",
                                 bounds_error=False)
        interped_amp = interp_amp(xx)

        if inplace:
            self.frq = xx
            self.amp = interped_amp
        else:
            return (xx, interped_amp)

    @classmethod
    def from_timeseries(cls, timeseries):
        """Compute the Fast Fourier Transform from a timeseries.

        Args:
            timeseries: TimeSeries 
                TimeSeries object to be transformed.

        Returns:
            An initialized FourierTransform object.
        """
        amp, frq = cls.fft(timeseries.amp, timeseries.dt)
        return cls(amp, frq, timeseries.fnyq)

    @property
    def mag(self):
        """Magnitude of complex fft amplitude."""
        return np.abs(self.amp)

    @property
    def phase(self):
        """Phase of complex fft amplitude in radians."""
        return np.angle(self.amp)

    @property
    def imag(self):
        """Imaginary component of complex fft amplitude."""
        return np.imag(self.amp)

    @property
    def real(self):
        """Real component of complex fft amplitude."""
        return np.real(self.amp)

    def __repr__(self):
        return f"FourierTransform(amp={str(self.amp[0:3])[:-1]} ... {str(self.amp[-3:])[1:]}, frq={str(self.frq[0:3])[:-1]} ... {str(self.frq[-3:])[1:]})"
