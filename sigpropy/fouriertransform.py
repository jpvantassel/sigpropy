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

"""This file contains the class FourierTransform."""

import numpy as np
import scipy.interpolate as sp
import scipy.fftpack as fftpack
# import numba

# Alternate implementation -> Does pass unit tests.
# @numba.jit(nopython=True, cache=True)
# def _k_and_o_smooth(amplitude, frequencies, bandwidth=40.0, normalize=True):
    
#     smooth_amp = np.empty_like(amplitude)
#     for c_id, cfrq in enumerate(frequencies):
#         if cfrq < 1E-6:
#             smooth_amp[c_id] = amplitude[c_id]
#             continue

#         total = 0
#         total_weight = 0
#         for comp_id, frequency in enumerate(frequencies):
#             if (frequency<1e-6):
#                 continue
#             elif np.abs(frequency - cfrq) < 1E-6:
#                 weight = 1
#             else:
#                 weight = bandwidth * np.log10(frequency/cfrq)
#                 weight = np.sin(weight)/weight
#                 weight = weight*weight*weight*weight

#             total += weight * amplitude[comp_id]
#             total_weight += weight

#         if total_weight > 0:
#             if normalize:
#                 smooth_amp[c_id] = total/total_weight
#             else:
#                 smooth_amp[c_id] = total
#         else:
#             smooth_amp[c_id] = 0
#     return smooth_amp

# Albert's implementation -> Doesnt pass unit tests.
# @numba.jit(nopython=True)
# def smooth(ko_freqs, freqs, spectrum, b):
#     max_ratio = pow(10.0, (3.0 / b))
#     min_ratio = 1.0 / max_ratio

#     total = 0
#     window_total = 0

#     ko_smooth = np.empty_like(ko_freqs)
#     for i, fc in enumerate(ko_freqs):
#         if fc < 1e-6:
#             ko_smooth[i] = 0
#             continue

#         total = 0
#         window_total = 0
#         for j, freq in enumerate(freqs):
#             frat = freq / fc

#             if (freq < 1e-6 or frat > max_ratio or frat < min_ratio):
#                 continue
#             elif np.abs(freq - fc) < 1e-6:
#                 window = 1.
#             else:
#                 x = b * np.log10(frat)
#                 window = np.sin(x) / x
#                 window *= window
#                 window *= window

#             total += window * spectrum[j]
#             window_total += window

#         if window_total > 0:
#             ko_smooth[i] = total / window_total
#         else:
#             ko_smooth[i] = 0

#     return ko_smooth

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
        self.amp = np.array(amplitude)
        self.frq = np.array(frequency)
        self.fnyq = fnyq if fnyq != None else np.max(self.frq)

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
                smooth_amp[cid] = np.dot(self.amp,smoothing_window)
        else:
            for c_col, cfrq in enumerate(self.frq):
                smoothing_window = self._k_and_o_window(self.frq, cfrq,
                                                        bandwidth=bandwidth)
                for c_row, c_amp in enumerate(self.amp):
                    smooth_amp[c_row, c_col] = np.dot(c_amp,smoothing_window)
        self.amp = smooth_amp

    # def smooth_konno_ohmachi(self, bandwidth=40.0):
    #     """Apply Konno and Ohmachi smoothing.

    #     Parameters
    #     ----------
    #     bandwidth : float, optional
    #         Width of smoothing window, default is 40.

    #     Returns
    #     -------
    #     None
    #         Modifies the internal attribute `amp` to equal the
    #         smoothed value of `mag`.
    #     """

    #     if len(self.amp.shape) == 1:
    #         self.amp = self._k_and_o_smooth(self.mag, self.frq, bandwidth=bandwidth) 
    #     else:
    #         for c_win, c_mag in enumerate(self.mag):
    #             self.amp[c_win] = self._k_and_o_smooth(c_mag, self.frq, bandwidth=bandwidth)

    # @staticmethod
    # def _k_and_o_smooth(amplitude, frequencies, bandwidth=40.0, normalize=True):
    #     return _k_and_o_smooth(amplitude, frequencies, bandwidth, normalize)
    #     # return smooth(frequencies, frequencies, amplitude, bandwidth)
            
    @staticmethod
    def _k_and_o_window(frequencies, center_frequency,
                        bandwidth=40.0, normalize=True):
        if frequencies.dtype != np.float32 and frequencies.dtype != np.float64:
            msg = 'frequencies needs to have a dtype of float32/64.'
            raise ValueError(msg)

        if center_frequency == 0:
            smoothing_window = np.zeros(
                len(frequencies), dtype=frequencies.dtype)
            smoothing_window[frequencies == 0.0] = 1.0
            return smoothing_window

        with np.errstate(divide='ignore', invalid='ignore'):
            smoothing_window = bandwidth * \
                np.log10(frequencies / center_frequency)
            smoothing_window = np.sin(smoothing_window) / smoothing_window
            smoothing_window = smoothing_window * \
                smoothing_window * smoothing_window * smoothing_window

        smoothing_window[frequencies == center_frequency] = 1.0
        smoothing_window[frequencies == 0.0] = 0.0

        if normalize:
            smoothing_window /= smoothing_window.sum()

        return smoothing_window

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
        return f"FourierTransform(amp={str(self.amp[0:3])[:-1]} ... {str(self.amp[-3:])[1:]}, frq={str(self.frq[0:3])[:-1]} ... {str(self.frq[-3:])[1:]})"
