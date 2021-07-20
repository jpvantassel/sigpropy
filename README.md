# sigpropy - A Python package for signal processing

> Joseph Vantassel, The University of Texas at Austin

[![DOI](https://zenodo.org/badge/218571161.svg)](https://zenodo.org/badge/latestdoi/218571161)
[![PyPI - License](https://img.shields.io/pypi/l/sigpropy)](https://github.com/jpvantassel/sigpropy/blob/main/LICENSE.txt)
[![CircleCI](https://circleci.com/gh/jpvantassel/sigpropy.svg?style=svg)](https://circleci.com/gh/jpvantassel/sigpropy)
[![Documentation Status](https://readthedocs.org/projects/sigpropy/badge/?version=latest)](https://sigpropy.readthedocs.io/en/latest/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/jpvantassel/sigpropy.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/jpvantassel/sigpropy/context:python)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/1a9c785e79214a0db457797f6d5f82f0)](https://www.codacy.com/manual/jpvantassel/sigpropy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jpvantassel/sigpropy&amp;utm_campaign=Badge_Grade)
[![codecov](https://codecov.io/gh/jpvantassel/sigpropy/branch/main/graph/badge.svg?token=GOR8BPD1PR)](https://codecov.io/gh/jpvantassel/sigpropy)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sigpropy)
[![Maintainability](https://api.codeclimate.com/v1/badges/6a3d16f9a406a5367a67/maintainability)](https://codeclimate.com/github/jpvantassel/sigpropy/maintainability)

## Table of Contents

-   [About _sigpropy_](#About-sigpropy)
-   [TimeSeries](#TimeSeries)
-   [FourierTransform](#FourierTransform)

## About _sigpropy_

_sigpropy_ is a Python package for digital signal processing. It includes two
main class definitions, _TimeSeries_ and _FourierTransform_. These classes
include methods to perform common signal processing techniques (e.g., trimming
and resampling) and properties to make using them readable and intuitive.

This package and the classes therein are being used in several other
Python projects, some of which have been released publicly and others are
still in the development stage, so if you do not see a feature you would like
it may very well be under development and released in the near future. To be
notified of future releases, you can either `watch` the repository on
[GitHub](https://github.com/jpvantassel/sigpropy) or
`Subscribe to releases` on the
[Python Package Index (PyPI)](https://pypi.org/project/sigpropy/).

## TimeSeries

A simple example:

```Python3
import sigpropy
import matplotlib.pyplot as plt
import numpy as np

dt = 0.002
time = np.arange(0, 1, dt)
s1 = 1*np.sin(2*np.pi*10*time)
s2 = 2*np.sin(2*np.pi*20*time)
s3 = 5*np.sin(2*np.pi*30*time)
amplitude = s1 + s2 + s3

tseries = sigpropy.TimeSeries(amplitude, dt)
fseries = sigpropy.FourierTransform.from_timeseries(tseries)

plt.plot(tseries.time, tseries.amplitude)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()
```

<img src="https://github.com/jpvantassel/sigpropy/blob/main/figs/example_tseries.png?raw=true" width="425" />

## FourierTransform

A simple example:

```Python3
import sigpropy
import matplotlib.pyplot as plt
import numpy as np

dt=0.002
time = np.arange(0, 1, dt)
s1 = 1*np.sin(2*np.pi*10*time)
s2 = 2*np.sin(2*np.pi*20*time)
s3 = 5*np.sin(2*np.pi*30*time)
amplitude = s1 + s2 + s3

tseries = sigpropy.TimeSeries(amplitude, dt)
fseries = sigpropy.FourierTransform.from_timeseries(tseries)

plt.plot(fseries.frequency, fseries.mag)
plt.xscale("log")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|FFT Amplitude|")
plt.show()
```

<img src="https://github.com/jpvantassel/sigpropy/blob/main/figs/example_fseries.png?raw=true" width="425" />

## Special Thanks To

- __Albert Kottke__ for his suggestions to speed up the Konno and Ohmachi
smoothing. For a standalone implementation of Konno and Ohmachi smoothing see
his project [pykooh](https://github.com/arkottke/pykooh).
