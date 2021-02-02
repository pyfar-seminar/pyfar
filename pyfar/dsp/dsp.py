import numpy as np
from scipy import signal as sgn
from pyfar import Signal
import pyfar.fft as fft
import pyfar.dsp.fractional_octave_smoothing as fs


def phase(signal, deg=False, unwrap=False):
    """Returns the phase for a given signal object.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the pyfar signal class
    deg : Boolean
        Specifies, whether the phase is returned in degrees or radians.
    unwrap : Boolean
        Specifies, whether the phase is unwrapped or not.
        If set to "360", the phase is wrapped to 2 pi.

    Returns
    -------
    phase : np.array()
        Phase.
    """

    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    phase = np.angle(signal.freq)

    if np.isnan(phase).any() or np.isinf(phase).any():
        raise ValueError('Your signal has a point with NaN or Inf phase.')

    if unwrap is True:
        phase = np.unwrap(phase)
    elif unwrap == '360':
        phase = wrap_to_2pi(np.unwrap(phase))

    if deg:
        phase = np.degrees(phase)
    return phase


def group_delay(signal, frequencies=None):
    """Returns the group delay of a signal in samples.

    Parameters
    ----------
    signal : Signal object
        An audio signal object from the pyfar signal class
    frequencies : number array like
        Frequency or frequencies in Hz at which the group delay is calculated.
        The default is None, in which case signal.frequencies is used.

    Returns
    -------
    group_delay : numpy array
        Frequency dependent group delay in samples. The array is flattened if
        a single channel signal was passed to the function.
    """

    # check input and default values
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    frequencies = signal.frequencies if frequencies is None \
        else np.asarray(frequencies)

    # get time signal and reshape for easy looping
    time = signal.time
    time = time.reshape((-1, signal.n_samples))
    # initialize group delay
    group_delay = np.zeros((np.prod(signal.cshape), frequencies.size))
    # calculate the group delay
    for cc in range(time.shape[0]):
        group_delay[cc] = sgn.group_delay(
            (time[cc], 1), frequencies, fs=signal.sampling_rate)[1]
    # reshape to match signal
    group_delay = group_delay.reshape(signal.cshape + (-1, ))

    # flatten in numpy fashion if a single channel is returned
    if signal.cshape == (1, ):
        group_delay = np.squeeze(group_delay)

    return group_delay


def wrap_to_2pi(x):
    """Wraps phase to 2 pi.

    Parameters
    ----------
    x : double
        Input phase to be wrapped to 2 pi.

    Returns
    -------
    x : double
        Phase wrapped to 2 pi.
    """
    positive_input = (x > 0)
    zero_check = np.logical_and(positive_input, (x == 0))
    x = np.mod(x, 2*np.pi)
    x[zero_check] = 2*np.pi
    return x


def nextpow2(x):
    """Returns the exponent of next higher power of 2.

    Parameters
    ----------
    x : double
        Input variable to determine the exponent of next higher power of 2.

    Returns
    -------
    nextpow2 : double
        Exponent of next higher power of 2.
    """
    return np.ceil(np.log2(x))


def spectrogram(signal, dB=True, log_prefix=20, log_reference=1,
                window='hann', window_length=1024, window_overlap_fct=0.5):
    """Compute the magnitude spectrum versus time.

    This is a wrapper for scipy.signal.spectogram with two differences. First,
    the returned times refer to the start of the FFT blocks, i.e., the first
    time is always 0 whereas it is window_length/2 in scipy. Second, the
    returned spectrogram is normalized accroding to `signal.signal_type` and
    `signal.fft_norm`.

    Parameters
    ----------
    signal : Signal
        pyfar Signal object.
    db : Boolean
        Falg to plot the logarithmic magnitude specturm. The default is True.
    log_prefix : integer, float
        Prefix for calculating the logarithmic time data. The default is 20.
    log_reference : integer
        Reference for calculating the logarithmic time data. The default is 1.
    window : str
        Specifies the window (See scipy.signal.get_window). The default is
        'hann'.
    window_length : integer
        Specifies the window length in samples. The default ist 1024.
    window_overlap_fct : double
        Ratio of points to overlap between fft segments [0...1]. The default is
        0.5

    Returns
    -------
    frequencies : numpy array
        Frequencies in Hz at which the magnitude spectrum was computed
    times : numpy array
        Times in seconds at which the magnitude spectrum was computed
    spectrogram : numpy array
    """

    # check input
    if not isinstance(signal, Signal):
        raise TypeError('Input data has to be of type: Signal.')

    if window_length > signal.n_samples:
        raise ValueError("window_length exceeds signal length")

    # get spectrogram from scipy.signal
    window_overlap = int(window_length * window_overlap_fct)
    window = sgn.get_window(window, window_length)

    frequencies, times, spectrogram = sgn.spectrogram(
            x=signal.time.squeeze(), fs=signal.sampling_rate, window=window,
            noverlap=window_overlap, mode='magnitude', scaling='spectrum')

    # remove normalization from scipy.signal.spectrogram
    spectrogram /= np.sqrt(1 / window.sum()**2)

    # apply normalization from signal
    spectrogram = fft.normalization(
        spectrogram, window_length, signal.sampling_rate,
        signal.fft_norm, window=window)

    # scipy.signal takes the center of the DFT blocks as time stamp we take the
    # beginning (looks nicer in plots, both conventions are used)
    times -= times[0]

    return frequencies, times, spectrogram


def fract_oct_smooth(src, smoothing_width, n_bins=None, phase_type=None):
    """
    Smooth magnitude spectrum of a signal with fractional octave width
    according to _[1]. If no signal is given, smoothing object is returned.

    Creates an object of class FractionalSmoothing to compute smoothing weights
    and to apply them on the input data. Returns a smoothed signal.

    To avoid boundery effect at the edge of the spectrum, the signal data is
    padded to fit the window size.
    See pyfar.fractional_octave_smoothing.data_padder and
    pyfar.fractional_octave_smoothing.apply for further information on the
    signal padding.

    Parameters
    ----------
    src : Signal
        Input signal to be smoothed
    smoothing_width : float, int
        Width of smoothing window relative to an octave
    n_bins : int, default None.
        Number of frequency bins of signal.
    phase_type : str, default None
        Phase handling specifier. `None` to return signal with zero phase.

    Returns
    -------
    Signal
        Smoothed Signal

    References
    ----------
    .. [1] J. G. Tylka, B. B. Boren, and E. Y. Choueiri,
           "A Generalized Method for Fractional-Octave Smoothing of Transfer
           Functions that Preserves Log-Frequency Symmetry,"
           J. Audio Eng. Soc., vol. 65, no. 3, pp. 239-245, (2017 March.).
           doi: https://doi.org/10.17743/jaes.2016.0053
    """
    if (src is None and n_bins is None) or \
       (src is not None and n_bins is not None):
        raise ValueError('Either signal or n_bins must be none.')

    if (src is not None and not isinstance(src, Signal)):
        raise TypeError("Input data must be of type Signal.")
    # Generate object
    obj = fs.FractionalSmoothing(src.n_bins, smoothing_width, phase_type)
    # If signal is given: return smoothed object
    if src is not None:
        return obj.apply(src)
    else:
        return obj
