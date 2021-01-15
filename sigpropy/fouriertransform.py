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

"""FourierTransform class definition."""

import logging

import numpy as np
import scipy.interpolate as sp
import scipy.fftpack as fftpack
from numba import njit

logger = logging.getLogger("sigpropy.fouriertransform")

__all__ = ['FourierTransform']


class FourierTransform():
    """A class for manipulating Fourier transforms.

    Attributes
    ----------
    frequency : ndarray
        Frequency vector of the transform in Hz.
    amplitude : ndarray
        The transform's amplitude in the same units as the input.
        May be 1D or 2D. If 2D each row corresponds to a unique FFT,
        where each column corresponds to an entry in `frequency`.
    fnyq : float
        The Nyquist frequency associated with the time series used
        to generate the Fourier transform. Note this may or may not
        be equal to `frequency[-1]`.

    """

    @staticmethod
    def fft(amplitude, dt, **kwargs):
        """Compute the fast-Fourier transform (FFT) of a time series.

        Parameters
        ----------
        amplitude : ndarray
            Denotes the time series amplitude. If `amplitude` is 1D
            each sample corresponds to a single time step. If
            `amplitude` is 2D each row corresponds to a particular
            section of the time record (i.e., time window) and each
            column corresponds to a single time step.
        dt : float
            Denotes the time step between samples in seconds.
        **kwargs : dict
            Additional keyard arguments to fft.

        Returns
        -------
        Tuple
            Of the form (frq, fft) where:

            frq : ndarray
                Positive frequency vector between zero and the
                Nyquist frequency (if even) or near the Nyquist
                (if odd) in Hz.

            fft : ndarray
                Complex amplitudes for the frequencies between zero
                and the Nyquist (if even) or near the Nyquist
                (if odd) with units of the input amplitude.
                If `amplitude` is a 2D array `fft` will also be a 2D
                array where each row is the FFT of each row of
                `amplitude`.

        """
        if amplitude.ndim > 2:
            raise TypeError("`amplitude` cannot have dimension > 2.")

        npts = amplitude.shape[-1] if kwargs.get("n") is None else kwargs.get("n")
        nfrqs = int(npts/2)+1 if (npts % 2) == 0 else int((npts+1)/2)
        frq = np.abs(np.fft.fftfreq(npts, dt))[:nfrqs]
        fft = 2/npts*fftpack.fft(amplitude, **kwargs)[:, :nfrqs]
        return (fft, frq)

    def __init__(self, amplitude, frequency, fnyq=None):
        """Initialize a `FourierTransform` object.

        Parameters
        ----------
        amplitude : ndarray
            Fourier transform amplitude. Refer to attribute `amp`
            for more details.
        frequency : ndarray
            Linearly spaced frequency vector for Fourier transform.
        fnyq : float, optional
            Nyquist frequency of Fourier transform, default is
            `max(frequency)`.

        Returns
        -------
        FourierTransform
            Initialized with `amplitude` and `frequency` information.

        """
        # amplitude must be castable to ndarray of complex doubles.
        try:
            self._amp = np.array(amplitude, dtype=np.cdouble)
        except TypeError or ValueError as e:
            msg = "`amplitude` must be convertable to `ndarray` of `cdouble`s."
            raise TypeError(msg) from e

        # amplitude must have ndim==2.
        if self._amp.ndim == 1:
            self._amp = np.expand_dims(self._amp, axis=0)
        elif self._amp.ndim > 2:
            msg = f"`amplitude` must be 1-D or 2-D, not {self._amp.ndim}-D."
            raise TypeError(msg)
        else:
            pass

        # frequency must be castable to ndarray of doubles.
        try:
            self._frq = np.array(frequency, dtype=np.double)
        except ValueError:
            msg = "`frequency` must be convertable to `ndarray` of `double`s."
            raise TypeError(msg)

        # frequency must have ndim=1.
        if self._frq.ndim != 1:
            msg = f"frequency must be 1-D, not {self._frq.ndim} 1-D."
            raise TypeError(msg)

        # fnyq (nyquist frequency) must be positive float.
        self.fnyq = float(fnyq) if fnyq is not None else float(max(self._frq))
        if self.fnyq <= 0:
            raise ValueError(f"fnyq must be greater than 0, not {self.fnyq}")

    @property
    def frequency(self):
        return self._frq

    @property
    def amplitude(self):
        return np.squeeze(self._amp)

    def smooth_konno_ohmachi(self, bandwidth=40.0):
        """Apply Konno and Ohmachi smoothing.

        Parameters
        ----------
        bandwidth : float, optional
            Width of smoothing window, default is 40.

        Returns
        -------
        None
            Modifies the internal attribute `amp` to equal the
            smoothed value of `mag`.

        """
        self._amp = self.mag
        smooth_amp = np.empty_like(self._amp, dtype=np.double)

        for c_col, cfrq in enumerate(self._frq):
            smoothing_window = self._k_and_o_window(self._frq, cfrq,
                                                    bandwidth=bandwidth)
            for c_row, c_amp in enumerate(self._amp):
                smooth_amp[c_row, c_col] = np.dot(c_amp, smoothing_window)

        self._amp = smooth_amp

    @staticmethod
    def _k_and_o_window(frequencies, center_frequency,
                        bandwidth=40.0, normalize=True):
        frequencies = np.array(frequencies, dtype=float)

        if center_frequency == 0:
            window = np.zeros_like(frequencies)
            window[frequencies == 0.0] = 1.0
            return window

        with np.errstate(divide='ignore', invalid='ignore'):
            window = bandwidth * np.log10(frequencies / center_frequency)
            window = np.sin(window) / window
            window = window * window * window * window

        window[frequencies == center_frequency] = 1.0
        window[frequencies == 0.0] = 0.0

        if normalize:
            window /= window.sum()

        return window

    def smooth_konno_ohmachi_fast(self, frequencies, bandwidth=40):
        """Apply fast Konno and Ohmachi smoothing.

        Parameters
        ----------
        frequencies : array-like
            Frequencies at which the smoothing is performed. If you
            choose to use all of the frequencies from the FFT for this
            parameter you should not expect
            much speedup over `smooth_konno_ohmachi`.
        bandwidth : float, optional
            Width of smoothing window, default is 40.

        Returns
        -------
        None
            Modifies the internal attribute `amp` to equal the
            smoothed value of `mag`.

        """
        frequencies = np.array(frequencies, dtype=np.double)

        self._amp = self._smooth_konno_ohmachi_fast(self._frq, self.mag,
                                                    fcs=frequencies,
                                                    bandwidth=bandwidth)
        self._frq = frequencies

    @staticmethod
    @njit(cache=True)
    def _smooth_konno_ohmachi_fast(frequencies, spectrum, fcs, bandwidth=40.):  # pragma: no cover
        """Static method for Konno and Ohmachi smoothing.

        Parameters
        ----------
        frequencies : ndarray
            Frequencies of the spectrum to be smoothed.
        spectrum : ndarray
            Spectrum to be smoothed, must be the same size as
            frequencies.
        fcs : ndarray
            Array of center frequencies where smoothed spectrum is
            calculated.
        bandwidth : float, optional
            Width of smoothing window, default is 40.

        Returns
        -------
        ndarray
            Spectrum smoothed at the specified center frequencies
            (`fcs`).

        """
        n = 3
        upper_limit = np.power(10, +n/bandwidth)
        lower_limit = np.power(10, -n/bandwidth)

        nrows = spectrum.shape[0]
        ncols = fcs.size
        smoothed_spectrum = np.empty((nrows, ncols))

        for fc_index, fc in enumerate(fcs):

            if fc < 1E-6:
                smoothed_spectrum[:, fc_index] = 0
                continue

            sumproduct = np.zeros(nrows)
            sumwindow = 0

            for f_index, f in enumerate(frequencies):
                f_on_fc = f/fc

                if (f < 1E-6) or (f_on_fc > upper_limit) or (f_on_fc < lower_limit):
                    continue
                elif np.abs(f - fc) < 1e-6:
                    window = 1.
                else:
                    window = bandwidth * np.log10(f_on_fc)
                    window = np.sin(window) / window
                    window *= window
                    window *= window

                sumproduct += window*spectrum[:, f_index]
                sumwindow += window

            if sumwindow > 0:
                smoothed_spectrum[:, fc_index] = sumproduct / sumwindow
            else:
                smoothed_spectrum[:, fc_index] = 0

        return smoothed_spectrum

    def resample(self, minf, maxf, nf, res_type="log", inplace=False):
        """Resample `FourierTransform` over a specified range.

        Parameters
        ----------
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
            if a copy is to be returned. By default the
            resampling is not done inplace (i.e., `inplace=False`).

        Returns
        -------
        None or Tuple
            If `inplace=True`
                `None`, method edits the internal attribute `amp`.
            If `inplace=False`
                A tuple of the form (`frequency`, `amplitude`)
                where `frequency` is the resampled frequency vector and
                `amplitude` is the resampled amplitude vector if
                `amp` is 1D or array if `amp` is 2D.

        Raises
        ------
        ValueError:
            If `maxf`, `minf`, or `nf` are illogical.
        NotImplementedError
            If `res_type` is not among those options specified.

        """
        if maxf < minf:
            raise ValueError("`maxf` must be > `minf`")
        if not isinstance(nf, int):
            raise TypeError("`nf` must be positive integer")
        if nf < 0:
            raise ValueError("`nf` must be positive integer")
        if maxf > self.fnyq*1.05:
            raise ValueError("`maxf` is out of range.")
        if minf < min(self._frq):
            raise ValueError("`minf` is out of range.")

        if res_type == "log":
            xx = np.geomspace(minf, maxf, nf)
        elif res_type == "linear":
            xx = np.linspace(minf, maxf, nf)
        else:
            msg = f"{res_type} resampling has not been implemented."
            raise NotImplementedError(msg)

        interp_amp = sp.interp1d(self._frq,
                                 self._amp,
                                 kind="linear",
                                 fill_value="extrapolate",
                                 bounds_error=False)
        interped_amp = interp_amp(xx)

        if inplace:
            self._frq = xx
            self._amp = interped_amp
        else:
            return (xx, interped_amp)

    @classmethod
    def from_timeseries(cls, timeseries, **fft_kwargs):
        """Create `FourierTransform` from `TimeSeries`.

        Parameters
        ----------
        timeseries : TimeSeries
            `TimeSeries` object to be transformed.
        **fft_kwargs : dict
            Custom settings for fft.

        Returns
        -------
        FourierTransform
            Initialized with information from `TimeSeries`.

        """
        amp, frq = cls.fft(timeseries._amp, timeseries.dt, **fft_kwargs)
        return cls(amp, frq, timeseries.fnyq)

    @property
    def mag(self):
        """Magnitude of complex FFT amplitude."""
        return np.abs(self._amp)

    @property
    def phase(self):
        """Phase of complex FFT amplitude in radians."""
        return np.angle(self._amp)

    @property
    def imag(self):
        """Imaginary component of complex FFT amplitude."""
        return np.imag(self._amp)

    @property
    def real(self):
        """Real component of complex FFT amplitude."""
        return np.real(self._amp)

    def __str__(self):
        """Human-readable representation of `FourierTransform`."""
        return f"FourierTransform of shape {self.amplitude.shape} at {id(self)}"

    def __repr__(self):
        """Unambiguous representation of `FourierTransform`."""
        return f"FourierTransform(amplitude=np.{self.amplitude.__repr__()}, frequency=np.{self.frequency.__repr__()}, fnyq={self.fnyq})"
