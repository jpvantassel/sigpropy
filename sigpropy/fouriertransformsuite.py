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

"""This file contains the class FourierTransformSuite."""

import numpy as np
import warnings
from sigpropy import FourierTransform, WindowedTimeSeries
from numba import njit

__all__ = ['FourierTransformSuite']


class FourierTransformSuite(FourierTransform):
    """A class for multiple Fourier transforms.

    Attributes
    ----------
    frequency : ndarray
        Frequency vector of the transform in Hz.
    amplitude : ndarray
        The transform's amplitude in the same units as the input. Each
        row corresponds to a unique FFT, where each column corresponds
        to an entry in `frequency`.
    fnyq : float
        The Nyquist frequency associated with the time series used
        to generate the Fourier transform. Note this may or may not
        be equal to `frequency[-1]`.
    """

    def __init__(self, amplitude, frequency, fnyq=None):
        """Initialize a `MultiFourierTransform` object.

        Parameters
        ----------
        amplitude : ndarray
            Fourier transform amplitude as 2D array, each row is a
            unique FFT, and each column correponds to a specific
            `frequency`.
        frequency : ndarray 
            Frequency vector, one per column of `amplitude`.
        fnyq : float, optional
            Nyquist frequency of Fourier transform, default is 
            `max(frq)`.

        Returns
        -------
        FourierTransformSuite
            Initialized with `amplitude` and `frequency` information.
        """
        super().__init__(amplitude, frequency, fnyq=fnyq)

    @staticmethod
    def _check_input(amplitude, frequency, fnyq):
        """Performs checks on input, specifically:

            1. Cast `amplitude` and `frequency` to ndarrays.
            2. Check that `amplitude` and `frequency` are 2D.
            3. Check that fnyq is greater than zero.

        """

        amplitude = np.array(amplitude)
        frequency = np.array(frequency)

        if len(frequency.shape) != 1:
            msg = f"`frequency` must be 2-D not {len(frequency.shape)}-D."
            raise ValueError(msg)

        if len(amplitude.shape) != 2:
            msg = f"`amplitude` must be 2-D not {len(amplitude.shape)}-D."
            raise ValueError(msg)
        
        if not fnyq > 0:
            raise ValueError(f"fnyq must be greater than 0, not {fnyq}")

        return amplitude, frequency, fnyq


    @staticmethod
    @njit()
    def _smooth_konno_ohmachi_fast(frequencies, spectrums, fcs,
                                   bandwidth=40):
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

        nrows = spectrums.shape[0]
        ncols = fcs.size
        smoothed_spectrum = np.empty((nrows, ncols))

        sumproduct = np.empty(nrows)

        for fc_index, fc in enumerate(fcs):

            if fc < 1E-6:
                smoothed_spectrum[:, fc_index] = 0
                continue

            sumproduct *= 0
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
                sumproduct += window*spectrums[:, f_index]
                sumwindow += window

            if sumwindow > 0:
                smoothed_spectrum[:, fc_index] = sumproduct / sumwindow
            else:
                smoothed_spectrum[:, fc_index] = 0

        return smoothed_spectrum

    def smooth_konno_ohmachi(self, bandwidth=40):
        msg = "smooth_konno_ohmachi will be removed in a future release"
        warnings.warn(msg, DeprecationWarning)
        super().smooth_konno_ohmachi(bandwidth=bandwidth)

    @classmethod
    def from_timeseries(cls, timeseries, **kwargs):
        """Create a `FourierTransform` object from a `TimeSeries`
        object.

        Parameters
        ----------
        timeseries : WindowedTimeSeries 
            `TimeSeries` object to be transformed.
        **kwargs : dict
            Custom settings for fft.

        Returns
        -------
        FourierTransformSuite
            Initialized with information from `WindowedTimeSeries`.

        Raises
        ------
        TypeError
            If `timeseries`, is not an instance of `WindowedTimeSeries`.

        """
        if not isinstance(timeseries, WindowedTimeSeries):
            msg = "Only valid for `WindowedTimeSeries`."
            raise TypeError(msg)
        return super().from_timeseries(timeseries, **kwargs)