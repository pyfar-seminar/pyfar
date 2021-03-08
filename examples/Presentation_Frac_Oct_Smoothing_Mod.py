# %%
from pyfar.dsp import fractional_octave_smoothing as fs
from pyfar.dsp import dsp as dsp
from pyfar import Signal
import numpy as np
import cProfile
import multiprocessing

from pyfar.testing import stub_utils
import pyfar.plot as plot
import matplotlib.pyplot as plt

# %% Smoothing object methods overview:
print(fs.FractionalSmoothing.__doc__)

for m in dir(fs.FractionalSmoothing):
    if not (m.startswith("_") or m.endswith("_")):
        func = getattr(fs.FractionalSmoothing, m)
        if hasattr(func, "__doc__"):
            print("- {:21s}{}".format(m + "()", func.__doc__.split("\n")[0]))

for m in dir(fs.PaddingType):
    if not (m.startswith("_") or m.endswith("_")):
        func = getattr(fs.PaddingType, m)
        if hasattr(func, "__doc__"):
            print("- {:21s}{}".format(m, func.__doc__.split("\n")[0]))

for m in dir(fs.PhaseType):
    if not (m.startswith("_") or m.endswith("_")):
        func = getattr(fs.PhaseType, m)
        if hasattr(func, "__doc__"):
            print("- {:21s}{}".format(m, func.__doc__.split("\n")[0]))

# %% # Source signal:
np.random.seed(1)
# White noise short
n_bins_short = 5000
c_shape = 1
src_magn_short = np.ones((c_shape, n_bins_short))
src_phase_short = np.random.random((c_shape, n_bins_short))
f_s = 44100
fft_norm = 'none'
white_noise_short = Signal(src_magn_short*np.exp(1j*2*np.pi*src_phase_short),
                           f_s,
                           domain='freq',
                           fft_norm=fft_norm)
shape = (1, 5000)
# White noise long
n_bins_long = 15000
src_magn_long = np.ones((c_shape, n_bins_long))
src_phase_long = np.random.random((c_shape, n_bins_long))
white_noise_long = Signal(src_magn_long*np.exp(1j*2*np.pi*src_phase_long),
                          f_s,
                          domain='freq',
                          fft_norm=fft_norm)
# Sine:
freq = 1000
n_samples = 44100
time_data, freq_data, frequency = stub_utils.sine_func(freq, f_s, n_samples,
                                                       fft_norm, (1,))
sine_1k = Signal(time_data, f_s, domain='time', fft_norm=fft_norm)

# %% Create smoothing object:
win_width = 1.3
phase_type = fs.PhaseType.ORIGINAL
paddig_type = fs.PaddingType.EDGE
smoother = fs.FractionalSmoothing(white_noise_short.n_bins,
                                  smoothing_width=win_width,
                                  phase_type=phase_type,
                                  padding_type=paddig_type)

print(smoother)

# %% Smooth 1k sine
smoothed_sine_1k_oct = dsp.fract_oct_smooth(src=sine_1k, smoothing_width=1,
                                            phase_type=fs.PhaseType.ORIGINAL,
                                            padding_type=fs.PaddingType.EDGE)

smoothed_sine_1k_terz = dsp.fract_oct_smooth(src=sine_1k, smoothing_width=1/3,
                                             phase_type=fs.PhaseType.ORIGINAL,
                                             padding_type=fs.PaddingType.EDGE)

fig = plt.figure()
plt.title('1 kHz Sine smoothed')
plot.line.freq(sine_1k, style='light', label='Original')
plot.line.freq(smoothed_sine_1k_oct, style='light', label='Octave')
plot.line.freq(smoothed_sine_1k_terz, style='light', label='Terz')
plt.legend()
plt.xlim([600, 1500])
# %% Time measurement
% % time
dsp.fract_oct_smooth(white_noise_short, win_width,
                     n_bins=None,
                     phase_type=phase_type,
                     padding_type=paddig_type)
# %% Time measurement with multiprocessing
pool = multiprocessing.Pool(processes=12)
# %%
% % time
pool.apply(dsp.fract_oct_smooth, args=(white_noise_short,
                                       win_width,
                                       None,
                                       phase_type,
                                       paddig_type))
# %%
% % time
pool.apply(dsp.fract_oct_smooth, args=(white_noise_long,
                                       win_width,
                                       None,
                                       phase_type,
                                       paddig_type))
pool.close()

# %%
cProfile.run(("dsp.fract_oct_smooth(white_noise_long, win_width, n_bins=None,"
              "phase_type=phase_type,"
              "padding_type=paddig_type)"))
# %% --------------------------------------------------------------------------
# Padding Settings:
# Zero padding
paddig_type = fs.PaddingType.ZERO
output_zero_padding = dsp.fract_oct_smooth(white_noise_short, win_width,
                                           n_bins=None,
                                           phase_type=phase_type,
                                           padding_type=paddig_type)
# Edge padding
paddig_type = fs.PaddingType.EDGE
output_edge_padding = dsp.fract_oct_smooth(white_noise_short, win_width,
                                           n_bins=None,
                                           phase_type=phase_type,
                                           padding_type=paddig_type)
# Mean padding
paddig_type = fs.PaddingType.MEAN
output_mean_padding = dsp.fract_oct_smooth(white_noise_short, win_width,
                                           n_bins=None,
                                           phase_type=phase_type,
                                           padding_type=paddig_type)

# Plot magnitude
fig = plt.figure()
plot.line.freq(white_noise_short, style='dark', label='Unsmoothed')
plot.line.freq(output_zero_padding, style='dark', label='Zero')
plot.line.freq(output_edge_padding, style='dark', label='Edge')
plot.line.freq(output_mean_padding, style='dark', label='Mean')
plt.legend()
plt.xlim([6000, 22050])
plt.ylim([-10, 5])
# %%
