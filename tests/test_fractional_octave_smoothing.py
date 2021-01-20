import pytest
from pytest import approx

import pyfar.dsp.fractional_octave_smoothing as fs
from pyfar import Signal
import numpy as np


@pytest.fixture
def smoother():
    signal_length = 100      # Signal length in freq domain
    win_width = 5
    # Create smoothing object
    smoother = fs.FractionalSmoothing(signal_length, win_width)
    # Compute integration limits
    limits = smoother.calc_integration_limits()
    # Compute weights:
    smoother.calc_weights()
    return smoother, limits


def test_init():
    signal_length = 10
    win_width = 1
    smoother = fs.FractionalSmoothing(signal_length, win_width)
    assert isinstance(smoother, fs.FractionalSmoothing)
    assert smoother._smoothing_width == win_width


def test_init_exceptions():
    signal_length = 10
    win_width = 1
    with pytest.raises(Exception) as error:
        assert fs.FractionalSmoothing('str', win_width)
    assert str(error.value) == "Invalid data type of number of bins (int)."
    with pytest.raises(Exception) as error:
        assert fs.FractionalSmoothing(signal_length, 'str')
    assert str(error.value) == "Invalid data type of window width (int/float)."


def test_calc_integration_limits(smoother):
    # 1.    Check if cutoff values are correct and at correct position
    # 2.    Check if sum of cutoff values == win_width
    # 3.    Check each limit between cutoff limits
    smoothing_obj, limits = smoother
    # Transform limits to dense matrix:
    limits = np.array([limits[0].toarray(), limits[1].toarray()])
    # Swap axis to make iteration easier:
    limits = limits.swapaxes(0, 1)
    # Window size
    win_width = smoothing_obj._smoothing_width
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
    signal_length = smoothing_obj._n_bins
    # Window size
    win_width = smoothing_obj._smoothing_width
    # Get weights:
    weights = (smoothing_obj._weights).toarray()
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


def test_apply():
    channels = 1
    signal_length = 30      # Signal length in freq domain
    data = np.zeros((channels, signal_length), dtype=np.complex)
    data[:, 3] = 1
    win_width = 3
    # Create smoothing object
    smoother = fs.FractionalSmoothing(signal_length, win_width)
    # Compute weights:
    smoother.calc_weights()
    # Apply
    smoothed_data = smoother.apply(data)
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
    data = np.empty((1, 1), dtype=np.complex128)
    f_s = 44100
    win_width = 1
    signal = Signal(data, f_s)
    smoothed_signal = fs.frac_smooth_signal(signal, win_width)
    assert isinstance(smoothed_signal, Signal)
    assert smoothed_signal._n_samples == signal._n_samples
    assert smoothed_signal._data.shape == signal._data.shape
