import pytest
import pyfar.dsp.fractional_octave_smoothing as fs


def test_Fractional_Smoothing_init():
    f_s = 44100
    win_width = 1
    win_type = 'rectangular'
    smoother = fs.FractionalSmoothing(f_s, win_width, win_type)
    assert isinstance(smoother, fs.FractionalSmoothing)
    assert smoother._sampling_rate == f_s
    assert smoother._smoothing_width == win_width
    assert smoother._win_type == win_type
