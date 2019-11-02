"""This file contains the class FourierTransform for creating and 
working with fourier transform objects."""

import numpy as np
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
import scipy.interpolate as sp


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

    def __init__(self, amplitude, frq):
        """Initialize a FourierTransform object.

        Args:
            amplitude: np.array
                Fourier transform amplitude.
            frq: np.array 
                Frequency vector for fourier transform

        Returns:
            An initialized FourierTransform object.
        """
        self.amp = amplitude
        self.frq = frq

    def smooth_konno_ohmachi(self, bandwidth=40.0, normalize=False):
        self.amp = konno_ohmachi_smoothing(self.mag, self.frq, bandwidth)

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

        if types[res_type] == types["log"]:
            xx = np.logspace(np.log10(minf), np.log10(maxf), nf)
        elif types[res_type] == types["linear"]:
            xx = np.linspace(minf, maxf, nf)
        else:
            raise NotImplementedError(
                f"{res_type} resampling has not been implemented.")

        interp_amp = sp.interp1d(self.frq, self.amp, kind="linear")
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
        return cls(amp, frq)

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
