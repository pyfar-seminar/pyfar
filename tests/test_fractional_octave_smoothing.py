import pytest
from pytest import approx

import pyfar.dsp.fractional_octave_smoothing as fs
import pyfar.dsp.dsp as dsp
from pyfar import Signal
import numpy as np


@pytest.fixture
def smoother():
    signal_length = 100      # Signal length in freq domain
    win_width = 5
    phase_type = 'Zero'
    # Create smoothing object
    smoother = fs.FractionalSmoothing(signal_length, win_width, phase_type)
    # Compute integration limits
    limits = smoother.calc_integration_limits()
    # Compute weights:
    smoother.calc_weights()
    return smoother, limits


def test_init():
    signal_length = 10
    win_width = 1
    phase_type = 'Zero'
    smoother = fs.FractionalSmoothing(signal_length, win_width, phase_type)
    assert isinstance(smoother, fs.FractionalSmoothing)
    assert smoother.smoothing_width == win_width


def test_init_exceptions():
    signal_length = 10
    win_width = 1
    with pytest.raises(Exception) as error:
        assert fs.FractionalSmoothing('str', win_width)
    assert str(error.value) == "Invalid data type of number of bins (int)."
    with pytest.raises(Exception) as error:
        assert fs.FractionalSmoothing(signal_length, 'str')
    assert str(error.value) == "Invalid data type of window width (int/float)."
    with pytest.raises(Exception) as error:
        assert fs.FractionalSmoothing(signal_length, win_width, 'str')
    assert str(error.value) == "Invalid phase type."


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
    # Get weights: (new_method)
    smoothing_obj.calc_weights_new()
    new_weights = (smoothing_obj._weights).toarray()
    # Check size of weighting matrix:
    assert weights.shape[0] == signal_length
    # Expected length of axis 1:
    # max_cutoff_bin = ceil(max_cutoff_value) + 1 (for k=0)
    expected_axis1_length = np.ceil((signal_length - 1)*2**(win_width/2)) + 1
    assert weights.shape[1] == int(expected_axis1_length)
    # Frequency bin k=0: no weights
    assert np.sum(weights[0]) == 0
    # Sum of weights for each bin == 1
    assert np.allclose(np.sum(weights[1:], axis=1),
                       np.full(signal_length-1, 1.),
                       atol=1e-16)
    assert np.allclose(weights, new_weights, atol=1e-16)

def test_weights_k10():
    signal_length = 100      # Signal length in freq domain
    # Check weight for k=10, win_width = 1
    win_width = 1
    # Create smoothing object
    smoother_k10 = fs.FractionalSmoothing(signal_length, win_width)
    # Compute weights:
    smoother_k10.calc_weights()
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


def test_apply():
    # Source signal:
    channels = 1
    signal_length = 30      # Signal length in freq domain
    data = np.zeros((channels, signal_length), dtype=np.complex)
    data[:, 3] = 1
    sampling_rate = 44100
    src_signal = Signal(data, sampling_rate, domain='freq')

    # Create smoothing object
    win_width = 1
    smoother = fs.FractionalSmoothing(signal_length, win_width)
    # Compute weights:
    smoother.calc_weights()
    # Apply
    smoothed_signal = smoother.apply(src_signal)
    smoothed_data = smoothed_signal.freq

    # Convert to array
    weights = (smoother._weights).toarray()
    weights[0, 0] = 1
    # Check shape
    assert smoothed_data.shape == data.shape

    # Pad data array for multiplication:
    # neglect padding with mean value
    # --> boundary effects if signal size too small
    pad_size_diff = weights.shape[-1] - data.shape[1]
    padded_data = np.pad(data[0], (0, pad_size_diff))

    # Check each freq bin:
    for k in range(smoothed_data.shape[1]):
        assert np.sum(weights[k, :]*padded_data) == smoothed_data[0, k]


# TODO
def test_smooth_signal():
    # Signal: 2kHz sine
    f_s = 100
    f_sin = 10
    dur = 1
    sine = np.atleast_2d(np.sin(2 * np.pi * f_sin * np.arange(dur*f_s) / f_s))
    src = Signal(sine, fs, n_samples=sine.size)
    # Kopie: src_copy = src oder src_copy = src.copy ?
    src_copy = src.copy()
    # Smoothind width
    win_width = 1
    smoothed_signal = dsp.fract_oct_smooth(src.copy(), win_width, phase_type='Zero')
    # Check if return type is correct:
    assert isinstance(smoothed_signal, Signal)
    # Check metadata:
    assert smoothed_signal._assert_matching_meta_data(src)
    # Check if input signal has changed:
    assert src == src_copy


def test_phase_handling():
    shape = (4, 9, 10)
    src_magn = np.ones(shape)
    src_phase = np.full(shape, np.pi)
    f_s = 44100
    win_width = 1
    signal = Signal(src_magn*np.exp(1j*src_phase), f_s, domain='freq')
    output_zero_phase = dsp.fract_oct_smooth(signal, win_width, n_bins=None,
                                             phase_type='Zero')
    output_orig_phase = dsp.fract_oct_smooth(signal, win_width, n_bins=None,
                                             phase_type='Original')
    assert np.array_equal(np.angle(output_zero_phase.freq), np.zeros(shape))
    assert np.array_equal(np.angle(output_orig_phase.freq), src_phase)
    with pytest.raises(Exception) as error:
        assert dsp.fract_oct_smooth(signal, win_width, n_bins=None,
                                    phase_type='Invalid')
    assert str(error.value) == "Invalid phase type."


def test_n_bins_setter(smoother):
    smoothing_obj, _ = smoother
    # Weights update should be False
    assert smoothing_obj._update_weigths is False
    # Initial number bins
    old_n_bins = smoothing_obj.n_bins
    # New number bins
    new_n_bins = 10*old_n_bins
    smoothing_obj.n_bins = new_n_bins
    assert smoothing_obj.n_bins == new_n_bins
    assert smoothing_obj._update_weigths is True


def test_smoothing_width_setter(smoother):
    smoothing_obj, _ = smoother
    # Weights update should be False
    assert smoothing_obj._update_weigths is False
    # Initial number bins
    old_smoothing_width = smoothing_obj.smoothing_width
    # New number bins
    new_smoothing_width = 10*old_smoothing_width
    smoothing_obj.smoothing_width = new_smoothing_width
    assert smoothing_obj.smoothing_width == new_smoothing_width
    assert smoothing_obj._update_weigths is True


def test_phase_setter_setter(smoother):
    smoothing_obj, _ = smoother
    # Weights update should be False
    assert smoothing_obj._update_weigths is False
    # Initial number bins
    old_phase_type = smoothing_obj.phase_type
    # New number bins
    new_phase_type = 'Original'
    smoothing_obj.phase_type = new_phase_type
    assert new_phase_type != old_phase_type
    assert smoothing_obj.phase_type == new_phase_type
    assert smoothing_obj._update_weigths is False
