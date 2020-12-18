import pytest
from pytest import approx

import pyfar.dsp.fractional_octave_smoothing as fs
from pyfar import Signal
import numpy as np


@pytest.fixture
def SetUpFractionalSmoothing():
    channel_number = 1
    signal_length = 10
    data = np.empty((channel_number, signal_length), dtype=np.complex)
    win_width = 1
    win_type = 'rectangular'
    return fs.FractionalSmoothing(data, win_width, win_type)


def test_init():
    channel_number = 1
    signal_length = 10
    data = np.empty((channel_number, signal_length), dtype=np.complex)
    win_width = 1
    smoother = fs.FractionalSmoothing(data, win_width)
    assert isinstance(smoother, fs.FractionalSmoothing)
    assert smoother._smoothing_width == win_width


def test_init_exceptions():
    data = np.empty((1, 1), dtype=np.complex128)
    wrong_ndarray_type = np.empty((1, 1), dtype=np.int)
    win_width = 1
    with pytest.raises(Exception) as error:
        assert fs.FractionalSmoothing('str', win_width)
    assert str(
        error.value) == "Invalid data type of input data (numpy.ndarray)."

    with pytest.raises(Exception) as error:
        assert fs.FractionalSmoothing(
            wrong_ndarray_type, win_width)
    assert str(
        error.value) == "ndarry must by of type: numpy.complex182."

    with pytest.raises(Exception) as error:
        assert fs.FractionalSmoothing(data, 'str', win_type)
    assert str(error.value) == "Invalid data type of window width (int/float)."


def DISABLED_test_calc_integration_limits():
    channel_number = 1
    signal_length = 5      # Signal length in freq domain
    data = np.empty((channel_number, signal_length), dtype=np.complex)
    win_width = 1
    win_type = 'rectangular'
    # Create smoothing object
    smoother = fs.FractionalSmoothing(data, win_width, win_type)
    # Compute integration limits
    smoother.calc_integration_limits()
    # Get integration_limits
    limits = smoother._limits

    # Check limits:
    # Freq bin zero:
    assert limits[0, :, :].all() == 0.0
    # Freq bin greater then zero:
    for k in range(1, limits.shape[0]):
        k_i = np.arange(limits.shape[2])
        # Compute cutoff bin of rectangular window:
        expected_lower_cutoff = k*2**(-win_width/2)
        expected_upper_cutoff = k*2**(win_width/2)
        # Indices of cutoff bins in array:
        # Last element that is smaller then lower cutoff
        idx_lower_cutoff = np.argmin((k_i - .5) < expected_lower_cutoff) - 1
        # If cutoff is larger than max signal size
        # -> last element is cutoff idx
        if expected_upper_cutoff > (k_i[-1]+.5):
            idx_upper_cutoff = k_i[-1]
        # Otherwise: first element that is larger then upper cutoff:
        else:
            idx_upper_cutoff = np.argmax(expected_upper_cutoff < (k_i + .5))

        # Limits at cutoff:
        expected_lower_limit = np.log2(expected_lower_cutoff/k)
        expected_upper_limit = np.log2(expected_upper_cutoff/k)

        # Limit at cutoff must equal log2(cutoff/idx_cutoff):
        # Upper limit in '0'
        assert limits[k, 0, int(idx_upper_cutoff)] == approx(
            expected_upper_limit)
        # Lower limit in '1'
        assert limits[k, 1, int(idx_lower_cutoff)] == approx(
            expected_lower_limit)

        # All limits smaller then cutoff must be zero:
        for k_i in range(int(idx_lower_cutoff)):
            assert limits[k, :, k_i] == approx(np.zeros(2))

        # All limits greater then cutoff must be zero:
        for k_i in range(int(idx_upper_cutoff)+1, limits.shape[2]):
            assert limits[k, :, k_i] == approx(np.zeros(2))

        # All limits in between:
        for k_i in range(int(idx_lower_cutoff), int(idx_upper_cutoff)):
            # Upper limit from idx_low to idx_upper - 1
            assert limits[k, 0, k_i] == np.log2((k_i+.5)/k)
            # Lower limit from idx_low + 1 to idx_upper + 1
            # Starts from k_i + 1
            assert limits[k, 1, k_i+1] == np.log2((k_i+1-.5)/k)


def DISABLED_test_smooth():
    data = np.empty((1, 1), dtype=np.complex128)
    f_s = 44100
    win_width = 1
    signal = Signal(data, f_s, signal_type='power')
    smoother = fs.FractionalSmoothing.smooth(signal, win_width)
    assert isinstance(smoother, fs.FractionalSmoothing)
    assert smoother._smoothing_width == win_width
    assert smoother._data == signal._data
    assert smoother._n_bins == signal._n_samples


def test_calc_integration_limits():
    # 1.    Check if cutoff values correct and at correct position
    # 2.    Check if sum of cutoff values == win_width
    # 3.    Check if diff between limits == 1
    channel_number = 1
    signal_length = 5      # Signal length in freq domain
    data = np.empty((channel_number, signal_length), dtype=np.complex)
    win_width = 1
    # Create smoothing object
    smoother = fs.FractionalSmoothing(data, win_width)
    # Compute integration limits
    smoother.calc_integration_limits()
    # Get integration_limits
    limits = smoother._limits

    # Check limits:
    # Freq bin zero:
    assert limits[0, :, :].all() == 0.0
    # Freq bin greater then zero:
    for k, limit in enumerate(limits[1:]):
        k += 1
        k_i = np.arange(limit.shape[1])
        # Compute cutoff bin of rectangular window:
        expected_lower_cutoff = k*2**(-win_width/2)
        expected_upper_cutoff = k*2**(win_width/2)
        # Indices of cutoff bins in array:
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
        assert limit[exp_i_upper_cutoff][0] + abs(
            limit[exp_i_low_cutoff][1]) == win_width
        assert 2**(limit[exp_i_low_cutoff+1:exp_i_upper_cutoff][0])*k - 2**(
            limit[exp_i_low_cutoff+1:exp_i_upper_cutoff][0])*k == 1
