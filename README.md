# SigProPy - A Python package for digital signal processing

> Joseph Vantassel, The University of Texas at Austin

## Table of Contents

- [About _sigpropy_](#About-sigpropy)
  - [TimeSeries](#TimeSeries)
  - [FourierTransform](#FourierTransform)

## About _sigpropy_

_sigpropy_ is a Python package for digital signal processing. It includes two
main class definitions, _TimeSeries_ and _FourierTransform_. These classes
include methods to perform common signal processing techniques (e.g., trimming
and resampling) and properties to make using them readable and inuitive. This
package and the classes therein are being used in several other
Python projects, some of which have been released publically and others are
still in the development stage, so if you do not see a feature you would like
it may very well be underdevelopment and released in the near future. To be
notified of future releases, you can either `watch` the repository on
[Github](https://github.com/jpvantassel/sigpropy) or
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

<img src="https://github.com/jpvantassel/sigpropy/blob/master/figs/example_tseries.png?raw=true" width="425">

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

<img src="https://github.com/jpvantassel/sigpropy/blob/master/figs/example_fseries.png?raw=true" width="425">
