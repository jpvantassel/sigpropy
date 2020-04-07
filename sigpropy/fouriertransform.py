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

import numpy as np
import scipy.interpolate as sp
import scipy.fftpack as fftpack
from numba import njit

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
            Additional keyard arguements to fft.

        Returns
        -------
        Tuple
            Of the form (frq, fft) where:

            frq : ndarray
                Positve frequency vector between zero and the
                Nyquist frequency (if even) or near the Nyquist
                (if odd) in Hz.

            fft : ndarray
                Complex amplitudes for the frequencies between zero
                and the Nyquist (if even) or near the Nyquist
                (if odd) with units of the input ampltiude.
                If `amplitude` is a 2D array `fft` will also be a 2D
                array where each row is the FFT of each row of
                `amplitude`.

        """
        if len(amplitude.shape) > 2:
            raise TypeError("`amplitude` cannot have dimension > 2.")

        npts = amplitude.shape[-1] if kwargs.get(
            "n") is None else kwargs.get("n")
        nfrqs = int(npts/2)+1 if (npts % 2) == 0 else int((npts+1)/2)
        frq = np.abs(np.fft.fftfreq(npts, dt))[0:nfrqs]
        if len(amplitude.shape) == 1:
            return(2/npts * fftpack.fft(amplitude, **kwargs)[0:nfrqs], frq)
        else:
            fft = np.zeros((amplitude.shape[0], nfrqs), dtype=complex)
            for cwindow, amplitude in enumerate(amplitude):
                fft[cwindow] = 2/npts * \
                    fftpack.fft(amplitude, **kwargs)[0:nfrqs]
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
            `max(frq)`.

        Returns
        -------
        FourierTransform
            Initialized with `amplitude` and `frequency` information.

        """
        fnyq = fnyq if fnyq != None else max(frequency)
        results = self._check_input(amplitude, frequency, fnyq)
        self.amp, self.frq, self.fnyq = results

    @staticmethod
    def _basic_checks(amplitude, frequency, fnyq):
        amplitude = np.array(amplitude)
        frequency = np.array(frequency)
        fnyq = float(fnyq)

        if len(frequency.shape) != 1:
            msg = f"Frequency must be 1-D not {len(frequency.shape)}-D."
            raise TypeError(msg)

        if not fnyq > 0:
            raise ValueError(f"fnyq must be greater than 0, not {fnyq}")

        return amplitude, frequency, fnyq

    def _check_input(self, amplitude, frequency, fnyq):
        """Performs checks on input, specifically:

        1. Check that `amplitude` and `frequency` are 1D.
        2. Checks that fnyq is greater than zero.

        """

        amplitude, frequency, fnyq = self._basic_checks(amplitude, frequency,
                                                        fnyq)

        if len(amplitude.shape) != 1:
            msg = f"Amplitude must be 1-D not {len(amplitude.shape)}-D."
            raise ValueError(msg)

        return amplitude, frequency, fnyq

    @property
    def frequency(self):
        return self.frq

    @property
    def amplitude(self):
        return self.amp

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
        self.amp = self.mag
        smooth_amp = np.empty_like(self.amp)
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
            choose to use all of the frequencies from the FFT
            (i.e., `self.frq`) for this parameter you should not expect
            much speedup over `smooth_konno_ohmachi`.
        bandwidth : float, optional
            Width of smoothing window, default is 40.

        Returns
        -------
        None
            Modifies the internal attribute `amp` to equal the
            smoothed value of `mag`.

        """
        frequencies = np.array(frequencies)
        self.amp = self._smooth_konno_ohmachi_fast(self.frequency, self.mag,
                                                   fcs=frequencies,
                                                   bandwidth=bandwidth)
        self.frq = frequencies

    @staticmethod
    @njit(cache=True)
    def _smooth_konno_ohmachi_fast(frequencies, spectrum, fcs, bandwidth=40):  # pragma: no cover
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

        smoothed_spectrum = np.empty_like(fcs)

        for f_index, fc in enumerate(fcs):
            if fc < 1E-6:
                smoothed_spectrum[f_index] = 0
                continue

            sumproduct = 0
            sumwindow = 0

            for f, c_spectrum in zip(frequencies, spectrum):
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
                sumproduct += window*c_spectrum
                sumwindow += window

            if sumwindow > 0:
                smoothed_spectrum[f_index] = sumproduct / sumwindow
            else:
                smoothed_spectrum[f_index] = 0

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
            If `res_type` is not amoung those options specified.

        """
        if maxf < minf:
            raise ValueError("`maxf` must be > `minf`")
        if not isinstance(nf, int):
            raise TypeError("`nf` must be postive integer")
        if nf < 0:
            raise ValueError("`nf` must be postive integer")
        if maxf > self.fnyq*1.05:
            raise ValueError("`maxf` is out of range.")
        if minf < min(self.frq):
            raise ValueError("`minf` is out of range.")

        if res_type == "log":
            xx = np.geomspace(minf, maxf, nf)
        elif res_type == "linear":
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
    def from_timeseries(cls, timeseries, **kwargs):
        """Create a `FourierTransform` object from a `TimeSeries`
        object.

        Parameters
        ----------
        timeseries : TimeSeries
            `TimeSeries` object to be transformed.
        **kwargs : dict
            Custom settings for fft.

        Returns
        -------
        FourierTransform
            Initialized with information from `TimeSeries`.

        """
        amp, frq = cls.fft(timeseries.amp, timeseries.dt, **kwargs)
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
        """Unambiguous representation of a `FourierTransform` object."""
        return f"FourierTransform(amp={str(self.amp[0:3])[:-1]} ... {str(self.amp[-3:])[1:]}, frq={str(self.frq[0:3])[:-1]} ... {str(self.frq[-3:])[1:]})"
