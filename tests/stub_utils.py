import numpy as np

from unittest import mock

from pyfar import Signal


def signal_stub(time, freq, sampling_rate, fft_norm):
    """Function to generate stub of pyfar Signal class based on MagicMock.
    The properties of the signal are set without any further check.

    Parameters
    ----------
    time : ndarray
        Time data
    freq : ndarray
        Frequency data
    sampling_rate : float
        Sampling rate
    fft_norm : 'unitary', 'amplitude', 'rms', 'power', 'psd'
        See documentaion of pyfar.fft.normalization.

    Returns
    -------
    signal
        stub of pyfar Signal class
    """

    # Use MagicMock and side_effect to mock __getitem__
    # See "Mocking a dictionary with MagicMock",
    # https://het.as.utexas.edu/HET/Software/mock/examples.html
    def getitem(slice):
        time = np.atleast_2d(signal.time[slice])
        freq = np.atleast_2d(signal.freq[slice])
        item = signal_stub(
                        time,
                        freq,
                        signal.sampling_rate,
                        signal.fft_norm)
        return item

    signal = mock.MagicMock(spec_set=Signal(
                                        time,
                                        sampling_rate,
                                        domain='time'))
    signal.time = np.atleast_2d(time)
    signal.freq = np.atleast_2d(freq)
    signal.sampling_rate = sampling_rate
    signal.fft_norm = fft_norm
    signal.n_samples = time.shape[-1]
    signal.n_bins = freq.shape[-1]
    signal.cshape = time.shape[:-1]
    signal.times = np.atleast_1d(
                        np.arange(0, signal.n_samples) / sampling_rate)
    signal.frequencies = np.atleast_1d(
                        np.fft.rfftfreq(signal.n_samples, sampling_rate))
    signal.__getitem__.side_effect = getitem

    return signal


def impulse_func(delay, n_samples, fft_norm, cshape):
    """ Generate time and frequency data of delta impulse.

    Parameters
    ----------
    n_samples : int
        Number of samples
    fft_norm : 'unitary', 'rms'
        See documentaion of pyfar.fft.normalization.
    delay  : ndarray, int
        Delay in samples
    cshape : tuple
        Channel shape

    Returns
    -------
    time : ndarray, float
        time vector
    freq : ndarray, complex
        Spectrum

    """
    # Convert delay to array
    delay = np.atleast_1d(delay)
    if np.shape(delay) != cshape:
        raise ValueError("Shape of delay needs to equal cshape.")
    if delay.max() >= n_samples:
        raise ValueError("Delay is larger than number of samples,"
                         f"which is {n_samples}")
    
    # Time vector
    time = np.zeros(cshape+(n_samples,))
    for idx, d in np.ndenumerate(delay):
        time[idx+(d,)] = 1
    # Spectrum
    n_bins = int(n_samples / 2) + 1
    bins = np.broadcast_to(np.arange(n_bins), (cshape+(n_bins,)))
    freq = np.exp(-1j * 2 * np.pi * bins * delay[..., np.newaxis] / n_samples)
    # Normalization
    freq = _normalization(freq, n_samples, fft_norm)

    return time, freq


# TO DO cshape
def sine_func(frequency, sampling_rate, n_samples, fft_norm):
    """ Generate time and frequency data of sine signal.
    The frequency is adjusted resulting in a fully periodic signal in the
    given time interval.

    Parameters
    ----------
    frequency : float
        Frequency of sine
    sampling_rate : float
        Sampling rate
    n_samples : int
        Number of samples
    fft_norm : 'none', 'rms'
        See documentaion of pyfar.fft.normalization.

    Returns
    -------
    time : ndarray, float
        time vector
    freq : ndarray, complex
        frequency vector
    frequency : float
        adjusted frequency

    """
    if frequency >= sampling_rate/2:
        raise ValueError(f"Frequency can be {sampling_rate/2} maximum,"
                         f"but is {frequency}")
    # Round to the nearest frequency bin
    n_periods = np.floor(n_samples / sampling_rate * frequency)
    frequency = n_periods * sampling_rate / n_samples

    # Time vector
    times = np.arange(0, n_samples) / sampling_rate
    time = np.atleast_2d(np.sin(2 * np.pi * frequency * times))
    # Spectrum
    n_bins = int(n_samples / 2) + 1
    freq = np.atleast_2d(np.zeros(n_bins, dtype=np.complex))
    freq_bin = int(frequency / sampling_rate * n_samples)
    freq[..., freq_bin] = -1j
    # Normalization
    freq = _normalization(freq, n_samples, fft_norm)
    return time, freq, frequency


def _normalization(freq, n_samples, fft_norm):
    """Normalized spectrum as defined in _[1],
    see documentaion of pyfar.fft.normalization.

    Parameters
    ----------
    freq : ndarray, complex
        frequency data
    n_samples : int
        Number of samples
    fft_norm : 'none', 'rms'
        See documentaion of pyfar.fft.normalization.

    Returns
    -------
    freq
        Normalized frequency data

    References
    ----------
    .. [1] J. Ahrens, C. Andersson, P. Höstmad, and W. Kropp, “Tutorial on
           Scaling of the Discrete Fourier Transform and the Implied Physical
           Units of the Spectra of Time-Discrete Signals,” Vienna, Austria,
           May 2020, p. e-Brief 600.
    """
    norm = np.ones_like(freq)
    if fft_norm == 'rms':
        # Equation 4 in Ahrens et al. 2020
        norm /= n_samples
        # Equation 8 and 10 in Ahrens et al. 2020
        if n_samples % 2 != 0:
            norm[1:] *= np.sqrt(2)
        else:
            norm[1:-1] *= np.sqrt(2)
    elif fft_norm != 'none':
        raise ValueError(("norm type must be 'none' or 'rms', "
                          f"but is '{fft_norm}'"))
    freq_norm = norm * freq
    return freq_norm

