# This file is part of SigProPy a module for digital signal processing
# in python.
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

"""This file contains the class `FourierTransform` for creating and 
working with Fourier transform objects."""

import numpy as np
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing
import scipy.interpolate as sp
import scipy.fftpack as fftpack
# from numba import jit

# @jit(nopython=True)


def _center_zero(frequencies):
    smoothing_window = np.zeros(len(frequencies), dtype=frequencies.dtype)
    smoothing_window[frequencies == 0.0] = 1.0
    return smoothing_window

# @jit(nopython=True)


def _parta(frequencies, center_frequency, bandwidth):
    return bandwidth * np.log10(frequencies / center_frequency)

# @jit(nopython=True)


def _partb(frequencies, center_frequency, bandwidth, smoothing_window):
    return np.sin(smoothing_window) / smoothing_window

# @jit(nopython=True)


def _partc(frequencies, center_frequency, bandwidth, smoothing_window):
    return smoothing_window * smoothing_window * smoothing_window * smoothing_window


def _makewindow(frequencies, center_frequency, bandwidth=40.0):
    with np.errstate(divide='ignore', invalid='ignore'):
        smoothing_window = _parta(frequencies, center_frequency, bandwidth)
        smoothing_window = _partb(
            frequencies, center_frequency, bandwidth, smoothing_window)
        smoothing_window = _partc(
            frequencies, center_frequency, bandwidth, smoothing_window)
    return smoothing_window


def _fix_window(frequencies, center_frequency, smoothing_window):
    smoothing_window[frequencies == center_frequency] = 1.0
    smoothing_window[frequencies == 0.0] = 0.0
    return smoothing_window


class FourierTransform():
    """A class for manipulating Fourier transforms.

    Attributes:
        frq : ndarray
            Frequency vector of the transform in Hz.
        amp : ndarray
            The transform's amplitude is in the same units as the input.
            May be 1D or 2D. If 2D each row corresponds to a unique FFT,
            where each column correpsonds to an entry in `frq`.
        fnyq : float
            The Nyquist frequency associated with the time series used
            to generate the Fourier transform. Note this may or may not
            be equal to `frq[-1]`.
    """

    @staticmethod
    def _check_input(name, values):
        """Perform simple checks on values of parameter `name`.

        Specifically:
            1. Check `values` is `ndarray`, `list`, or `tuple`. 
            2. If `list` or `tuple` convert to a `ndarray`.

        Args:
            name : str
                `name` of parameter to be check. Only used to raise 
                easily understood exceptions.
            values : any
                value of parameter to be checked.

        Returns:
            `values` as an `ndarray`.

        Raises:
            TypeError:
                If entries do not comply with checks 1. and 2. listed 
                above.
        """
        if type(values) not in [np.ndarray, list, tuple]:
            msg = f"{name} must be of type ndarray, list, or tuple not {type(values)}."
            raise TypeError(msg)

        if isinstance(values, (list, tuple)):
            values = np.array(values)

        return values

    @staticmethod
    def fft(amplitude, dt):
        """Compute the fast-Fourier transform (FFT) of a time series.

        Args:
            amplitude : ndarray
                Denotes the time series amplitude. If `amplitude` is 1D
                each sample corresponds to a single time step. If
                `amplitude` is 2D each row corresponds to a particular
                section of the time record (i.e., time window) and each
                column corresponds to a single time step.
            dt : float
                Denotes the time step between samples in seconds.

        Returns:
            Tuple of the form (frq, fft) where:
                `frq` : ndarray
                    Positve frequency vector between zero and the
                    Nyquist frequency (if even) or near the Nyquist
                    (if odd) in Hz.
                `fft` : ndarray
                    Complex amplitudes for the frequencies between zero
                    and the Nyquist (if even) or near the Nyquist 
                    (if odd) with units of the input ampltiude.
                    If `amplitude` is a 2D array `fft` will also be a 2D
                    array where each row is the FFT of each row of 
                    `amplitude`.
        """
        if len(amplitude.shape) > 2:
            raise TypeError("`amplitude` cannot have dimension > 2.")

        npts = amplitude.shape[-1]
        nfrqs = int(npts/2)+1 if (npts % 2) == 0 else int((npts+1)/2)
        frq = np.abs(np.fft.fftfreq(npts, dt))[0:nfrqs]
        if len(amplitude.shape) == 1:
            # return(2/npts * np.fft.fft(amplitude)[0:nfrqs], frq)
            return(2/npts * fftpack.fft(amplitude)[0:nfrqs], frq)
        else:
            fft = np.zeros((amplitude.shape[0], nfrqs), dtype=complex)
            for cwindow, amplitude in enumerate(amplitude):
                # fft[cwindow] = 2/npts * np.fft.fft(amplitude)[0:nfrqs]
                fft[cwindow] = 2/npts * fftpack.fft(amplitude)[0:nfrqs]
            return (fft, frq)

    def __init__(self, amplitude, frq, fnyq=None):
        """Initialize a `FourierTransform` object.

        Args:
            amplitude : ndarray
                Fourier transform amplitude. Refer to attribute `amp`
                for more details.
            frq : ndarray 
                Linearly spaced frequency vector for Fourier transform.
            fnyq : float, optional
                Nyquist frequency of Fourier transform, default is 
                `max(frq)`.

        Returns:
            An initialized `FourierTransform` object.
        """
        # TODO (jpv): Remove inconsistency between freq and amp.
        self.amp = FourierTransform._check_input("amplitude", amplitude)
        self.frq = FourierTransform._check_input("frq", frq)
        self.fnyq = fnyq if fnyq != None else np.max(self.frq)

    def smooth_konno_ohmachi(self, bandwidth=40.0):
        """ Apply Konno and Ohmachi smoothing.

        Args:
            bandwidth : float, optional
                Width of smoothing window, default is 40.

        Returns:
            `None`, modifies the internal attribute `amp` to equal the
            smoothed value of `mag`.
        """
        # self.amp = konno_ohmachi_smoothing(self.mag, self.frq, bandwidth,
        #                                    enforce_no_matrix=False, max_memory_usage=2048,
        #                                    normalize=True)
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
                        bandwidth=40.0, normalize=True):
        if frequencies.dtype != np.float32 and frequencies.dtype != np.float64:
            msg = 'frequencies needs to have a dtype of float32/64.'
            raise ValueError(msg)

        if center_frequency == 0:
            return _center_zero(frequencies)

        smoothing_window = _makewindow(frequencies, center_frequency, bandwidth)

        smoothing_window = _fix_window(
            frequencies, center_frequency, smoothing_window)

        if normalize:
            smoothing_window /= smoothing_window.sum()

        return smoothing_window

    def resample(self, minf, maxf, nf, res_type="log", inplace=False):
        """Resample `FourierTransform` over a specified range.

        Args:
            minf : float 
                Minimum value of resample.
            maxf : float
                Maximum value of resample.
            nf : int
                Number of resamples.
            res_type : {"log", "linear"}, optional
                Type of resampling, default value is `log`.
            inplace : bool, optional
                Determines whether resampling is done in place or 
                if a copy is returned be returned. By default the
                resampling is not done inplace (i.e., `inplace=False`).

        Returns:
            If `inplace=True`
                `None`, method edits the internal attribute `amp`.
            If `inplace=False`
                A tuple of the form (`frequency`, `amplitude`)
                where `frequency` is the resampled frequency vector and 
                `amplitude` is the resampled amplitude vector if 
                `amp` is 1D or array if `amp` is 2D. 

        Raises:
            ValueError:
                If `maxf`, `minf`, or `nf` are illogical.
            NotImplementedError:
                If `res_type` is not amoung those options specified.
        """
        if maxf < minf:
            raise ValueError("`maxf` must be > `minf`")
        if type(nf) not in (int,):
            raise TypeError("`nf` must be postive integer")
        if nf < 0:
            raise TypeError("`nf` must be postive integer")
        types = {"log": "log", "linear": "linear"}
        if maxf > self.fnyq*1.05:
            raise ValueError("`maxf` is out of range.")
        if minf < 0:
            raise ValueError("`minf` is out of range.")

        if types[res_type] == types["log"]:
            xx = np.logspace(np.log10(minf), np.log10(maxf), nf)
        elif types[res_type] == types["linear"]:
            xx = np.linspace(minf, maxf, nf)
        else:
            msg = f"{res_type} resampling has not been implemented."
            raise NotImplementedError(msg)

        interp_amp = sp.interp1d(self.frq,
                                 self.amp,
                                 kind="linear",
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
        """Create a `FourierTransform` object from a `TimeSeries`
        object.

        Args:
            timeseries : TimeSeries 
                `TimeSeries` object to be transformed.

        Returns:
            An initialized `FourierTransform` object.
        """
        amp, frq = cls.fft(timeseries.amp, timeseries.dt)
        return cls(amp, frq, timeseries.fnyq)

    @property
    def mag(self):
        """Magnitude of complex FFT amplitude."""
        return np.abs(self.amp)

    @property
    def phase(self):
        """Phase of complex FFT amplitude in radians."""
        return np.angle(self.amp)

    @property
    def imag(self):
        """Imaginary component of complex FFT amplitude."""
        return np.imag(self.amp)

    @property
    def real(self):
        """Real component of complex FFT amplitude."""
        return np.real(self.amp)

    def __repr__(self):
        return f"FourierTransform(amp={str(self.amp[0:3])[:-1]} ... {str(self.amp[-3:])[1:]}, frq={str(self.frq[0:3])[:-1]} ... {str(self.frq[-3:])[1:]})"
