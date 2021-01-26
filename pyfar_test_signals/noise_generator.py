"""Generate colored noise.

USES MIT LICENSED FILE colorednoise.py from
felixpatzelt / colorednoise
https://raw.githubusercontent.com/felixpatzelt/colorednoise/master/colorednoise.py

Contributors:

@felixpatzelt
Felix Patzelt
@atspaeth
Alex Spaeth

Part of:
pyfar_test_signals

Copyright ©2021 Jonas Oertel

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

from numpy import sqrt, newaxis
from numpy.fft import irfft, rfftfreq
from numpy.random import normal
from numpy import sum as npsum
import numpy as np
import pyfar
from pyfar import Signal    # managing audio signals

# Standard settings
fs = 44100


def noise():
    # 3 Noise was chosen
    # Confirmation
    print('You chose noise.\n')
    # Ask which kind of noise should be generated
    noise_color = int(input('Which kind of noise would you like to generate?\n'
                            'Please choose the corresponding number:\n'
                            '1: White noise\n'
                            '2: Pink noise\n'
                            '3: Brown noise\n'))
    # Set exponent (beta) for colorednoise.py according to choice
    if noise_color == 1:
        exponent = 0  # White noise
    elif noise_color == 2:
        exponent = 1  # Pink noise
    elif noise_color == 3:
        exponent = 2  # Brown noise
    else:
        print('Invalid choice, sorry! :\\')
    # Ask to choose duration in seconds
    L = float(input('Choose duration in seconds:\n'))
    # Confirmation
    print('The duration will be ', L, ' s.')
    samples_length = L*fs
    np.round_(samples_length)
    samples_length = int(samples_length)
    size = samples_length

    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to

        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2

        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.

    Returns
    -------
    out : array
        The samples.


    Examples:
    ---------

    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we assume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Build scaling factors for all frequencies
    s_scale = f
    fmin = 0
    fmin = max(fmin, 1./samples)  # Low frequency cutoff
    ix = npsum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.  # correct f = +-0.5
    sigma = 2 * sqrt(npsum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

    # Generate scaled random power + phase
    sr = normal(scale=s_scale, size=size)
    si = normal(scale=s_scale, size=size)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2): si[..., -1] = 0

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0

    # Combine power + corrected phase to Fourier components
    s = sr + 1J * si

    # Transform to real time series & scale to unit variance
    noise = irfft(s, n=samples, axis=-1) / sigma

    noise_power = Signal(noise, fs, signal_type='power')
    # Ask for desired filename (for wav)
    filename = input('Choose filename:\n'
                     '(If already existing, file will be overwritten!)\n')
    # Write the file
    pyfar.io.write_wav(noise_power, filename, overwrite=True)
    # Return signal
    return noise_power
