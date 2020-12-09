import pytest
import pyfar.dsp.fractional_octave_smoothing as fs

from pyfar import Signal
import numpy as np

# @pytest.fixture
# def setUp():
#     f_s = 44100
#     win_width = 1
#     win_type = 'rectangular'
#     return fs.FractionalSmoothing(f_s, win_width, win_type)


def test_init():
    data = np.empty((1, 1), dtype=np.complex)
    win_width = 1
    win_type = 'rectangular'
    smoother = fs.FractionalSmoothing(data, win_width, win_type)
    assert isinstance(smoother, fs.FractionalSmoothing)
    assert smoother._smoothing_width == win_width
    assert smoother._win_type == win_type


def test_init_exceptions():
    data = np.empty((1, 1), dtype=np.complex128)
    wrong_ndarray_type = np.empty((1, 1), dtype=np.int)
    win_width = 1
    win_type = 'rectangular'
    with pytest.raises(Exception) as error:
        assert fs.FractionalSmoothing('str', win_width, win_type)
    assert str(
        error.value) == "Invalid data type of input data (numpy.ndarray)."

    with pytest.raises(Exception) as error:
        assert fs.FractionalSmoothing(
            wrong_ndarray_type, win_width, win_type)
    assert str(
        error.value) == "ndarry must by of type: numpy.complex182."

    with pytest.raises(Exception) as error:
        assert fs.FractionalSmoothing(data, 'str', win_type)
    assert str(error.value) == "Invalid data type of window width (int/float)."

    with pytest.raises(Exception) as error:
        assert fs.FractionalSmoothing(data, win_width, 'hanning')
    assert str(error.value) == "Not a valid window type ('rectangular')."


def test_from_signal():
    data = np.empty((1, 1), dtype=np.complex128)
    f_s = 44100
    win_width = 1
    win_type = 'rectangular'
    signal = Signal(data, f_s, signal_type='power')
    smoother = fs.FractionalSmoothing.from_signal(signal, win_width, win_type)
    assert isinstance(smoother, fs.FractionalSmoothing)
    assert smoother._smoothing_width == win_width
    assert smoother._win_type == win_type
    assert smoother._data == signal._data
    assert smoother._n_bins == signal._n_samples
