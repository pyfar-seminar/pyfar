import numpy as np
import numpy.testing as npt
import pytest

from pyfar import Signal
from pyfar import fft


def test_signal_init(sine):
    """Test to init Signal without optional parameters."""
    signal = Signal(sine, 44100, domain='time')
    assert isinstance(signal, Signal)


def test_signal_init_list(impulse_list):
    signal = Signal(impulse_list, 44100, domain='time')
    assert isinstance(signal, Signal)


def test_signal_init_default_parameter(impulse_list):
    # using all defaults
    signal = Signal(impulse_list, 44100)
    assert signal.domain == 'time'
    assert signal.fft_norm == 'none'
    assert signal.comment is None

    # default of fft_norm depending on signal type
    signal = Signal(impulse_list, 44100)
    assert signal.fft_norm == 'none'


def test_signal_comment():
    signal = Signal([1, 0, 0], 44100, comment='Bla')
    assert signal.comment == 'Bla'


def test_domain_getter_freq(sine):
    signal = Signal(np.array([1]), 44100)
    signal._domain = 'freq'
    assert signal.domain == 'freq'


def test_domain_getter_time(sine):
    signal = Signal(np.array([1]), 44100)
    signal._domain = 'time'
    assert signal.domain == 'time'


def test_domain_setter_error():
    signal = Signal(np.array([1]), 44100)
    with pytest.raises(ValueError, match='Incorrect domain'):
        signal.domain = 'quark'


def test_domain_setter_freq_when_freq(sine):
    signal = Signal(np.array([1]), 44100)
    domain = 'freq'
    signal._domain = domain
    signal.domain = domain
    assert signal.domain == domain


def test_domain_setter_freq_when_time(sine):
    fft_norm = 'rms'
    samplingrate = 40e3
    spec = np.atleast_2d(fft.rfft(sine, len(sine), samplingrate, fft_norm))
    signal = Signal(sine, 44100, domain='time', fft_norm=fft_norm)
    domain = 'freq'
    signal.domain = domain
    assert signal.domain == domain
    npt.assert_allclose(signal._data, spec, atol=1e-14, rtol=1e-14)


def test_domain_setter_time_when_time(sine):
    signal = Signal(np.array([1]), 44100)
    domain = 'time'
    signal._domain = domain
    signal.domain = domain
    assert signal.domain == domain


def test_domain_setter_time_when_freq(sine):
    fft_norm = 'rms'
    samplingrate = 40e3
    spec = np.atleast_2d(fft.rfft(sine, len(sine), samplingrate, fft_norm))
    signal = Signal(spec, 44100, domain='freq', fft_norm=fft_norm)
    signal._data = spec
    signal._n_samples = len(sine)
    domain = 'time'
    signal.domain = domain
    assert signal.domain == domain
    npt.assert_allclose(
        signal._data, np.atleast_2d(sine), atol=1e-14, rtol=1e-14)


def test_signal_init_val(sine):
    """Test to init Signal with complete parameters."""
    signal = Signal(sine, 44100, domain="time", fft_norm='rms')
    assert isinstance(signal, Signal)


def test_signal_init_false_coord(sine):
    """Test to init Signal with position that is not of type Coordinates."""
    coord_false = np.array([1, 1, 1])
    with pytest.raises(TypeError):
        Signal(sine, 44100, position=coord_false)
        pytest.fail("Input value has to be coordinates object.")


def test_n_samples(impulse):
    """Test for number of samples."""
    data = impulse
    signal = Signal(data, 44100, domain='time')
    assert signal.n_samples == len(data)


def test_n_bins(sine):
    """Test for number of freq bins."""
    data = sine
    signal = Signal(data, 44100, domain='time')
    data_freq = np.fft.rfft(data)
    assert signal.n_bins == len(data_freq)


def test_times(sine):
    """Test for the time instances."""
    signal = Signal(sine, 44100, domain='time')
    times = np.atleast_1d(np.arange(0, len(sine)) / 44100)
    npt.assert_allclose(signal.times, times)


def test_getter_time(sine, impulse):
    """Test if attribute time is accessed correctly."""
    signal = Signal(sine, 44100)
    signal._domain = 'time'
    signal._data = impulse
    npt.assert_allclose(signal.time, impulse)


def test_setter_time(sine, impulse):
    """Test if attribute time is set correctly."""
    signal = Signal(sine, 44100)
    signal.time = impulse
    assert signal._domain == 'time'
    npt.assert_allclose(np.atleast_2d(impulse), signal._data)


def test_getter_freq(sine, impulse):
    """Test if attribute freq is accessed correctly."""
    samplingrate = 44100
    signal = Signal(sine, samplingrate, fft_norm='rms')
    new_sine = sine * 2
    spec = fft.rfft(new_sine, len(new_sine), samplingrate, 'rms')
    signal._domain = 'freq'
    signal._data = spec
    npt.assert_allclose(signal.freq, spec, atol=1e-15)


def test_setter_freq(sine, impulse):
    """Test if attribute freq is set correctly."""
    samplingrate = 44100
    signal = Signal(sine, samplingrate)
    spec = fft.rfft(impulse, len(impulse), samplingrate, fft_norm='unitary')
    signal.freq = spec
    assert signal.domain == 'freq'
    npt.assert_allclose(np.atleast_2d(spec), signal._data, atol=1e-15)


def test_getter_sampling_rate(sine):
    """Test if attribute sampling rate is accessed correctly."""
    sampling_rate = 48000
    signal = Signal(sine, 44100)
    signal._sampling_rate = sampling_rate
    npt.assert_allclose(signal.sampling_rate, sampling_rate)


def test_setter_sampligrate(sine):
    """Test if attribute sampling rate is set correctly."""
    sampling_rate = 48000
    signal = Signal(sine, 44100)
    signal.sampling_rate = sampling_rate
    npt.assert_allclose(sampling_rate, signal._sampling_rate)


def test_getter_signal_type(sine):
    """Test if attribute signal type is accessed correctly."""
    signal_type = "energy"
    signal = Signal(sine, 44100, fft_norm='none')
    signal._signal_type = signal_type
    npt.assert_string_equal(signal.signal_type, signal_type)

    signal_type = "energy"
    signal = Signal(sine, 44100, fft_norm='rms')
    signal._signal_type = signal_type
    npt.assert_string_equal(signal.signal_type, signal_type)


def test_setter_signal_type(sine):
    """Test if attribute signal type is set correctly."""
    signal_type = "energy"
    signal = Signal(sine, 44100)
    with pytest.raises(DeprecationWarning):
        signal.signal_type = signal_type


def test_getter_fft_norm(sine):
    signal = Signal(sine, 44100, fft_norm='psd')
    assert signal.fft_norm == 'psd'


def test_setter_fft_norm(sine):
    spec_power_unitary = np.atleast_2d([1, 2, 1])
    spec_power_amplitude = np.atleast_2d([1/4, 2/4, 1/4])

    signal = Signal(
        spec_power_unitary, 44100, n_samples=4, domain='freq',
        fft_norm='unitary')

    # changing the fft_norm also changes the spectrum
    signal.fft_norm = 'amplitude'
    assert signal.fft_norm == 'amplitude'
    npt.assert_allclose(signal.freq, spec_power_amplitude, atol=1e-15)

    # changing the fft norm in the time domain does not change the time data
    signal.domain = 'time'
    time_power_amplitude = signal._data.copy()
    signal.fft_norm = 'unitary'
    npt.assert_allclose(signal.time, time_power_amplitude)
    npt.assert_allclose(signal.freq, spec_power_unitary)

    # setting an invalid fft_norm
    with pytest.raises(ValueError):
        signal.fft_norm = 'bullshit'


def test_dtype(sine):
    """Test for the getter od dtype."""
    dtype = np.float64
    signal = Signal(sine, 44100, dtype=dtype)
    assert signal.dtype == dtype


def test_signal_length(sine):
    """Test for the signal length."""
    signal = Signal(sine, 44100)
    length = (1000 - 1) / 44100
    assert signal.signal_length == length


def test_cshape(sine, impulse):
    """Test the attribute cshape."""
    data = np.array([sine, impulse])
    signal = Signal(data, 44100)
    assert signal.cshape == (2,)


def test_magic_getitem(sine, impulse):
    """Test slicing operations by the magic function __getitem__."""
    data = np.array([sine, impulse])
    sr = 44100
    signal = Signal(data, sr)
    npt.assert_allclose(Signal(sine, sr)._data, signal[0]._data)


def test_magic_getitem_slice(sine, impulse):
    """Test slicing operations by the magic function __getitem__."""
    data = np.array([sine, impulse])
    sr = 44100
    signal = Signal(data, sr)
    npt.assert_allclose(Signal(sine, sr)._data, signal[:1]._data)


def test_magic_getitem_allslice(sine, impulse):
    """Test slicing operations by the magic function __getitem__."""
    data = np.array([sine, impulse])
    sr = 44100
    signal = Signal(data, sr)
    npt.assert_allclose(Signal(data, sr)._data, signal[:]._data)


def test_magic_setitem(sine, impulse):
    """Test the magic function __setitem__."""
    sr = 44100
    signal = Signal(sine, sr)
    set_signal = Signal(sine*2, sr)
    signal[0] = set_signal
    npt.assert_allclose(signal._data, set_signal._data)


def test_magic_setitem_wrong_sr(sine, impulse):
    """Test the magic function __setitem__."""
    sr = 44100
    signal = Signal(sine, sr)
    set_signal = Signal(sine*2, 48000)
    with pytest.raises(ValueError, match='sampling rates do not match'):
        signal[0] = set_signal


def test_magic_setitem_wrong_norm(sine, impulse):
    """Test the magic function __setitem__."""
    sr = 44100
    signal = Signal(impulse, sr, fft_norm='none')
    set_signal = Signal(sine*2, sr, fft_norm='rms')
    with pytest.raises(ValueError, match='FFT norms do not match'):
        signal[0] = set_signal


def test_magic_setitem_wrong_n_samples(sine, impulse):
    """Test the magic function __setitem__."""
    sr = 44100
    signal = Signal(sine, sr)
    set_signal = Signal(sine[..., :-10]*2, sr)
    with pytest.raises(ValueError, match='number of samples does not match'):
        signal[0] = set_signal


def test_magic_len(impulse):
    """Test the magic function __len__."""
    signal = Signal(impulse, 44100)
    assert len(signal) == 1000


def test_find_nearest_time():
    sampling_rate = 100
    signal = Signal(np.zeros(100), sampling_rate)
    actual = signal.find_nearest_time(0.5)
    expected = 50
    assert actual == expected

    actual = signal.find_nearest_time([0.5, 0.75])
    expected = [50, 75]
    npt.assert_allclose(actual, expected)


def test_find_nearest_frequency():
    sampling_rate = 100
    signal = Signal(np.zeros(100*2), sampling_rate*2)
    actual = signal.find_nearest_frequency(50)
    expected = 50
    assert actual == expected

    actual = signal.find_nearest_frequency([50, 75])
    expected = [50, 75]
    npt.assert_allclose(actual, expected)


def test_reshape():

    # test reshape with tuple
    signal_in = Signal(np.random.rand(6, 256), 44100)
    signal_out = signal_in.reshape((3, 2))
    npt.assert_allclose(signal_in._data.reshape(3, 2, -1), signal_out._data)
    assert id(signal_in) != id(signal_out)

    signal_out = signal_in.reshape((3, -1))
    npt.assert_allclose(signal_in._data.reshape(3, 2, -1), signal_out._data)
    assert id(signal_in) != id(signal_out)

    # test reshape with int
    signal_in = Signal(np.random.rand(3, 2, 256), 44100)
    signal_out = signal_in.reshape(6)
    npt.assert_allclose(signal_in._data.reshape(6, -1), signal_out._data)
    assert id(signal_in) != id(signal_out)


def test_reshape_exceptions():
    signal_in = Signal(np.random.rand(6, 256), 44100)
    signal_out = signal_in.reshape((3, 2))
    npt.assert_allclose(signal_in._data.reshape(3, 2, -1), signal_out._data)
    # test assertion for non-tuple input
    with pytest.raises(ValueError):
        signal_out = signal_in.reshape([3, 2])

    # test assertion for wrong dimension
    with pytest.raises(ValueError, match='Can not reshape signal of cshape'):
        signal_out = signal_in.reshape((3, 4))


def test_flatten():

    # test 2D signal (flatten should not change anything)
    x = np.random.rand(2, 256)
    signal_in = Signal(x, 44100)
    signal_out = signal_in.flatten()

    npt.assert_allclose(signal_in._data, signal_out._data)
    assert id(signal_in) != id(signal_out)

    # test 3D signal
    x = np.random.rand(3, 2, 256)
    signal_in = Signal(x, 44100)
    signal_out = signal_in.flatten()

    npt.assert_allclose(signal_in._data.reshape((6, -1)), signal_out._data)
    assert id(signal_in) != id(signal_out)


@pytest.fixture
def sine():
    """Generate a sine signal with f = 440 Hz and sampling_rate = 44100 Hz.

    Returns
    -------
    signal : ndarray, double
        The sine signal

    """
    amplitude = 1
    frequency = 440
    sampling_rate = 44100
    num_samples = 1000
    fullperiod = False

    if fullperiod:
        num_periods = np.floor(num_samples / sampling_rate * frequency)
        # round to the nearest frequency resulting in a fully periodic sine
        # signal in the given time interval
        frequency = num_periods * sampling_rate / num_samples
    times = np.arange(0, num_samples) / sampling_rate
    signal = amplitude * np.sin(2 * np.pi * frequency * times)

    return signal


def impulse_func():
    """Generate an impulse, also known as the Dirac delta function

    .. math::

        s(n) =
        \\begin{cases}
        a,  & \\text{if $n$ = 0} \\newline
        0, & \\text{else}
        \\end{cases}

    Returns
    -------
    signal : ndarray, double
        The impulse signal

    """
    amplitude = 1
    num_samples = 1000

    signal = np.zeros(num_samples, dtype=np.double)
    signal[0] = amplitude

    return signal


@pytest.fixture
def impulse():
    return impulse_func()


@pytest.fixture
def impulse_list():
    imp = impulse_func()

    return imp.tolist()
