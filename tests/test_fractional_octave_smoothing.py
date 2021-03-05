import pytest
from pytest import approx

from pyfar.dsp.fractional_octave_smoothing import FractionalSmoothing as fs
from pyfar.dsp.fractional_octave_smoothing import PhaseType
from pyfar.dsp.fractional_octave_smoothing import PaddingType
from pyfar.dsp import dsp as dsp
from pyfar import Signal
import numpy as np


@pytest.fixture
def smoother():
    signal_length = 100      # Signal length in freq domain
    win_width = 5
    phase_type = PhaseType.ZERO
    padding_type = PaddingType.MEAN
    # Create smoothing object
    smoother = fs(signal_length, win_width, phase_type,
                  padding_type=padding_type)
    # Compute integration limits
    limits = smoother._calc_integration_limits()
    # Compute weights:
    smoother._calc_weights()
    return smoother, limits


@pytest.fixture
def sine_2():
    # Source signal:
    sampling_rate = 44100
    frequency = 441
    dur = 0.01
    fft_norm = 'none'
    cshape = 2
    data = np.sin(2*np.pi*frequency*np.arange(dur *
                                              sampling_rate) / sampling_rate)
    data = np.expand_dims(data, axis=0)
    data = np.repeat(data, cshape, axis=0)
    return Signal(data, sampling_rate, domain='time', fft_norm=fft_norm)


@pytest.fixture
def white_noise_1():
    # Source signal:
    np.random.seed(1)
    shape = (1, 2205)
    src_magn = np.ones(shape)
    src_phase = np.random.random(shape)
    f_s = 44100
    fft_norm = 'none'
    return Signal(src_magn*np.exp(1j*2*np.pi*src_phase), f_s, domain='freq',
                  fft_norm=fft_norm)


def test_init():
    signal_length = 10
    win_width = 1
    phase_type = PhaseType.ZERO
    smoother = fs(signal_length, win_width, phase_type)
    assert isinstance(smoother, fs)
    assert smoother.smoothing_width == win_width


def test_init_exceptions():
    signal_length = 10
    win_width = 1

    with pytest.raises(Exception) as error:
        assert fs('str', win_width)
    assert str(error.value) == "Invalid data type of number of bins (int)."
    with pytest.raises(Exception) as error:
        assert fs(signal_length, 'str')
    assert str(error.value) == "Invalid data type of window width (int/float)."
    with pytest.raises(Exception, match="Invalid phase type."):
        assert fs(signal_length, win_width,
                  phase_type='str')

    with pytest.raises(Exception, match="Invalid padding type."):
        assert fs(signal_length, win_width,
                  padding_type='str')


def test_calc_integration_limits(smoother):
    # 1.    Check if cutoff values are correct and at correct position
    # 2.    Check if sum of cutoff values == win_width
    # 3.    Check each limit between cutoff limits
    smoothing_obj, limits = smoother
    # Transform limits to dense matrix:
    limits = np.array([limits[0].toarray(), limits[1].toarray()])
    # Swap axis to make iteration over k freq bins easier:
    limits = limits.swapaxes(0, 1)
    # Window size
    win_width = smoothing_obj.smoothing_width
    # Check limits:
    # Freq bin zero:
    assert limits[0, :, :].all() == 0.0
    # Freq bin greater then zero:
    for k, limit in enumerate(limits[1:], start=1):
        # Compute cutoff bin of rectangular window:
        expected_lower_cutoff = k*2**(-win_width/2)
        expected_upper_cutoff = k*2**(win_width/2)
        # Indices of cutoff bins in array:
        k_i = np.arange(limit.shape[1])
        # Last element that is smaller then lower cutoff
        exp_i_low_cutoff = np.argmin(
            (k_i - .5) < expected_lower_cutoff) - 1
        # First element that is larger then upper cutoff:
        exp_i_upper_cutoff = np.argmax(
            expected_upper_cutoff < (k_i + .5))

        # Limits at cutoff in scaled log:
        expected_lower_limit = np.log2(expected_lower_cutoff/k)
        expected_upper_limit = np.log2(expected_upper_cutoff/k)
        # Check value at expected position
        assert limit[1][exp_i_low_cutoff] == expected_lower_limit
        assert limit[0][exp_i_upper_cutoff] == expected_upper_limit
        # Check if sum of upper and lower limit equal win_width:
        assert limit[0][exp_i_upper_cutoff] + abs(
            limit[1][exp_i_low_cutoff]) == approx(win_width)

        for k_ii in range(exp_i_low_cutoff+1, exp_i_upper_cutoff):
            # Upper limit:
            assert limit[0][k_ii] == np.log2((k_ii+.5)/k)
            # Lower limit:
            assert limit[1][k_ii] == np.log2((k_ii-.5)/k)


def test_calc_weights(smoother):
    smoothing_obj, _ = smoother
    # Signal length
    signal_length = smoothing_obj.n_bins
    # Window size
    win_width = smoothing_obj.smoothing_width
    # Get weights:
    weights = (smoothing_obj._weights).toarray()
    # Check size of weighting matrix:
    assert weights.shape[0] == signal_length
    # Expected length of axis 1:
    # max_cutoff_bin = ceil(max_cutoff_value) + 1 (for k=0)
    expected_axis1_length = np.ceil((signal_length - 1)*2**(win_width/2)+1)
    assert weights.shape[1] == int(expected_axis1_length)
    # Frequency bin k= 0 and k'= 0:
    assert weights[0, 0] == 1
    # Sum of weights for each bin == 1
    # TODO:
    # Sum for weights[1] and weights[2] do not equal 1, becuause of
    # missing intervall from phi(k'_min) to phi(0.5)
    assert np.allclose(np.sum(weights, axis=1),
                       np.ones(signal_length),
                       atol=1e-16)


def test_weights_k10():
    signal_length = 100      # Signal length in freq domain
    # Check weight for k=10, win_width = 1
    win_width = 1
    # Create smoothing object
    smoother_k10 = fs(signal_length, win_width)
    # Compute weights:
    smoother_k10._calc_weights()
    weights_k10 = (smoother_k10._weights).toarray()
    k = 10
    # k*2^(-win_width) <= k' <= k*2^(win_width/2):
    k_min = k*2**(-win_width/2)
    k_max = k*2**(win_width/2)
    k_i = np.arange(int(np.floor(k_min)), int(np.ceil(k_max)))
    # phi(k,k' +/- .5) = log_2((k' +/- .5)/k) (Eq. 17)
    phi_up = np.log2((k_i+.5)/k)
    phi_low = np.log2((k_i-.5)/k)
    # Replace lowest/highest phi by phi_min/phi_max:
    phi_up[-1] = np.log2(k_max/k)
    phi_low[0] = np.log2(k_min/k)
    # Weights W[k=10, k'] = (phi_up - phi_low)/win_width
    # (Integral over rectangular window)
    expected_weights_k10 = (phi_up - phi_low)/win_width
    for i, w in enumerate(expected_weights_k10):
        assert w == weights_k10[10, i+k_i[0]]


def test_apply_via_matrix(sine_2):
    # Source signal:
    src_signal = sine_2

    # Create smoothing object with differing n_bins
    win_width = 1
    padding_type = PaddingType.ZERO
    smoother = fs(int(1.1*src_signal.n_bins), win_width,
                  padding_type=padding_type)

    # Apply with wrong input type:
    with pytest.raises(Exception) as error:
        assert smoother.apply_via_matrix('string')
    assert str(error.value) == "Invalid src input type (Signal)."

    # Apply with valid input type but differing n_bins:
    with pytest.raises(Exception) as error:
        assert smoother.apply_via_matrix(src_signal)
    assert str(error.value) == ("Input signal must have same number of "
                                "frequencies bins as set in smoothing object. "
                                "Set number of frequencies with obj.n_bins.")

    # Change n_bins:
    smoother.n_bins = src_signal.n_bins
    # Apply with valid input type and correct n_bins:
    smoothed_signal = smoother.apply_via_matrix(src_signal)

    # Convert to array
    weights = (smoother._weights).toarray()
    weights[0, 0] = 1
    # Check shape
    assert smoothed_signal.freq.shape == src_signal.freq.shape

    # Pad data array for multiplication:
    # neglect padding with mean value
    # --> boundary effects if signal size too small
    pad_width = weights.shape[-1] - src_signal.freq.shape[1]
    padded_magn = np.pad(np.abs(src_signal.freq), ((0, 0), (0, pad_width)))

    # Check each freq bin on first channel:
    for k in range(smoothed_signal.freq.shape[1]):
        assert np.sum(weights[k, :]*padded_magn[0, :]
                      ) == smoothed_signal.freq[0, k]


def test_apply_via_loop(sine_2):
    # Source signal:
    src_signal = sine_2

    # Create smoothing object with differing n_bins
    win_width = 1
    padding_type = PaddingType.EDGE
    smoother = fs(int(1.1*src_signal.n_bins), win_width,
                  padding_type=padding_type)

    # Apply with wrong input type:
    with pytest.raises(Exception) as error:
        assert smoother.apply('string')
    assert str(error.value) == "Invalid src input type (Signal)."

    # Apply with valid input type but differing n_bins:
    with pytest.raises(Exception) as error:
        assert smoother.apply(src_signal)
    assert str(error.value) == ("Input signal must have same number of "
                                "frequencies bins as set in smoothing object. "
                                "Set number of frequencies with obj.n_bins.")

    # Change n_bins:
    smoother.n_bins = src_signal.n_bins
    # Use old apply to check smoothed output:
    exp_smoothed_data = smoother.apply_via_matrix(src_signal).freq

    # Apply with valid input type and correct n_bins:
    smoothed_signal = smoother.apply(src_signal)
    smoothed_data = smoothed_signal.freq

    # Check shape
    assert smoothed_signal.freq.shape == src_signal.freq.shape
    # Check data
    assert np.allclose(smoothed_data, exp_smoothed_data, atol=1e-15)


def test_smooth_signal(sine_2):
    # Source signal:
    src_signal = sine_2
    src_copy = sine_2.copy()
    # Smoothind width
    win_width = 1
    padding_type = PaddingType.EDGE
    phase_type = PhaseType.ZERO
    smoothed_signal = dsp.fract_oct_smooth(src_signal, win_width,
                                           phase_type=phase_type,
                                           padding_type=padding_type)
    # Check if return type is correct:
    assert isinstance(smoothed_signal, Signal)
    # Check metadata:
    assert (smoothed_signal.n_samples == src_signal.n_samples) is True
    assert (smoothed_signal.sampling_rate == src_signal.sampling_rate) is True
    assert (smoothed_signal.fft_norm == src_signal.fft_norm) is True
    # Check if data of input signal has changed:
    assert np.alltrue(src_signal.time == src_copy.time)


def test_phase_handling():
    # Source signal
    shape = (4, 9, 10)
    src_magn = np.ones(shape)
    src_phase = np.full(shape, np.pi)
    f_s = 44100
    src_signal = Signal(src_magn*np.exp(1j*src_phase), f_s, domain='freq')

    # Smoothing objects
    win_width = 1
    padding_type = PaddingType.EDGE
    output_zero_phase = fs(src_signal.n_bins, win_width,
                           phase_type=PhaseType.ZERO,
                           padding_type=padding_type).apply(src_signal)
    output_orig_phase = fs(src_signal.n_bins, win_width,
                           phase_type=PhaseType.ORIGINAL,
                           padding_type=padding_type).apply(src_signal)

    # Check zero phase
    assert np.array_equal(np.angle(output_zero_phase.freq), np.zeros(shape))
    # Check original phase
    assert np.array_equal(np.angle(output_orig_phase.freq), src_phase)

    # Check error exceptions:
    with pytest.raises(Exception) as error:
        assert fs(src_signal.n_bins, win_width,
                  phase_type=PhaseType.MINIMUM,
                  padding_type=padding_type).apply(src_signal)
    assert str(error.value) == "PhaseType.MINIMUM is not implemented."
    with pytest.raises(Exception) as error:
        assert fs(src_signal.n_bins, win_width,
                  phase_type=PhaseType.LINEAR,
                  padding_type=padding_type).apply(src_signal)
    assert str(error.value) == "PhaseType.LINEAR is not implemented."
    with pytest.raises(Exception) as error:
        assert fs(src_signal.n_bins, win_width,
                  phase_type='invalid type',
                  padding_type=padding_type).apply(src_signal)
    assert str(error.value) == "Invalid phase type."


def test_padding_handling(white_noise_1):
    # Source signal
    src_signal = white_noise_1

    # Smoothing objects
    n_bins = src_signal.n_bins
    win_width = 1
    phase_type = PhaseType.ORIGINAL
    # Zero padding
    paddig_type = PaddingType.ZERO
    assert isinstance(fs(n_bins, win_width, phase_type=phase_type,
                         padding_type=paddig_type).apply(src_signal), Signal)
    # Edge padding
    paddig_type = PaddingType.EDGE
    assert isinstance(fs(n_bins, win_width, phase_type=phase_type,
                         padding_type=paddig_type).apply(src_signal), Signal)
    # Mean padding
    paddig_type = PaddingType.MEAN
    assert isinstance(fs(n_bins, win_width, phase_type=phase_type,
                         padding_type=paddig_type).apply(src_signal), Signal)
    # Exceptions:
    with pytest.raises(Exception) as error:
        assert fs(n_bins, win_width,
                  phase_type=phase_type,
                  padding_type="string").apply(src_signal)
    assert str(error.value) == "Invalid padding type."


def test_setter_n_bins(smoother):
    smoothing_obj, _ = smoother
    # Weights update should be True
    assert smoothing_obj._update_weigths is True
    # Initial number bins
    old_n_bins = smoothing_obj.n_bins
    # New number bins
    new_n_bins = 10*old_n_bins
    smoothing_obj.n_bins = new_n_bins
    assert smoothing_obj.n_bins == new_n_bins
    assert smoothing_obj._update_weigths is True
    # Invalid data type:
    with pytest.raises(Exception) as error:
        smoothing_obj.n_bins = 'string'
    assert str(error.value) == "Invalid data type of number of bins (int)."


def test_setter_smoothing_width(smoother):
    smoothing_obj, _ = smoother
    # Weights update should be True
    assert smoothing_obj._update_weigths is True
    # Initial smoothing width
    old_smoothing_width = smoothing_obj.smoothing_width
    # New smoothing width
    new_smoothing_width = 10*old_smoothing_width
    smoothing_obj.smoothing_width = new_smoothing_width
    assert smoothing_obj.smoothing_width == new_smoothing_width
    assert smoothing_obj._update_weigths is True
    # Invalid data type:
    with pytest.raises(Exception) as error:
        smoothing_obj.smoothing_width = 'string'
    assert str(error.value) == "Invalid data type of window width (int/float)."


def test_setter_phase(smoother):
    smoothing_obj, _ = smoother
    # Weights update should be True
    assert smoothing_obj._update_weigths is True
    # Initial phase type
    old_phase_type = smoothing_obj.phase_type
    # New phase type
    new_phase_type = PhaseType.ORIGINAL
    smoothing_obj.phase_type = new_phase_type
    assert new_phase_type != old_phase_type
    assert smoothing_obj.phase_type == new_phase_type
    assert smoothing_obj._update_weigths is True
    # Invalid phase type:
    with pytest.raises(Exception) as error:
        smoothing_obj.phase_type = 'string'
    assert str(error.value) == "Invalid phase type."


def test_setter_padding(smoother):
    smoothing_obj, _ = smoother
    # Weights update should be True
    assert smoothing_obj._update_weigths is True
    # Old padding type
    old_padding_type = smoothing_obj.padding_type
    # New padding type
    new_padding_type = PaddingType.EDGE
    smoothing_obj.padding_type = new_padding_type
    assert new_padding_type != old_padding_type
    assert smoothing_obj.padding_type == new_padding_type
    # Invalid padding type:
    with pytest.raises(Exception) as error:
        smoothing_obj.padding_type = 'string'
    assert str(error.value) == "Invalid padding type."


def test_data_padder(smoother):
    length = 10
    cshape = 3
    data = np.arange(length, dtype=np.float64)
    data = np.expand_dims(data, axis=0)
    data = np.repeat(data, cshape, axis=0)
    pad_width = 5
    mean_size = np.array([1, 3, 10])
    smoothing_obj, _ = smoother

    # MEAN
    padded_mean = smoothing_obj._data_padder(data, pad_width, mean_size,
                                             PaddingType.MEAN)
    assert (padded_mean.shape == (cshape, cshape, int(length+pad_width))) \
        is True
    assert np.array_equal(padded_mean[0, :, :], padded_mean[1, :, :]) is True
    assert np.array_equal(padded_mean[0, :, :], padded_mean[2, :, :]) is True
    assert np.alltrue(padded_mean[0, 0, length:] == np.mean(
        data[0, (length-mean_size[0]):]))
    assert np.alltrue(padded_mean[0, 1, length:] == np.mean(
        data[1, (length-mean_size[1]):]))
    assert np.alltrue(padded_mean[0, 2, length:] == np.mean(
        data[2, (length-mean_size[2]):]))

    # ZERO
    padded_zero = smoothing_obj._data_padder(data, pad_width, mean_size,
                                             PaddingType.ZERO)
    assert (padded_zero.shape == (cshape, cshape, int(length+pad_width))) \
        is True
    assert np.alltrue(padded_zero[0, :, length:] == 0)

    # EDGE
    padded_edge = smoothing_obj._data_padder(data, pad_width, mean_size,
                                             PaddingType.EDGE)
    assert (padded_zero.shape == (cshape, cshape, int(length+pad_width))) \
        is True
    assert np.alltrue(padded_edge[0, :, length:] == data[0, -1])

    # Exceptions:
    data_3d = np.atleast_3d(np.empty((10)))
    with pytest.raises(Exception) as error:
        assert smoothing_obj._data_padder(data_3d, pad_width, mean_size,
                                          PaddingType.MEAN)
    assert str(error.value) == "Data array must be 2D."
    with pytest.raises(Exception) as error:
        assert smoothing_obj._data_padder(data, pad_width, mean_size,
                                          'string')
    assert str(
        error.value) == "Invalid PaddingType or PaddingType not implemented."
