"""Generate a sine sweep.

Part of:
pyfar_test_signals
"""
import numpy as np
import pyfar
from pyfar import Signal    # managing audio signals


def sweep():
    # Standard settings
    fs = 44100
    # 4 Sine sweep was chosen
    # Confirmation
    print('You chose a sine sweep.')
    # Ask for type of sweep
    sweep_type = int(input('Choose type of sweep:\n'
                           '1: linear\n'
                           '2: logarithmic\n'))
    # Ask for duration of sweep
    T = float(input('Choose sweep duration in seconds:\n'))
    # Ask for start frequency
    f_start = float(input('Choose start frequency:\n'))
    # Ask for stop frequency
    f_stop = float(input('Choose stop frequency:\n'))
    # Confirm frequencies
    print('Your sine sweep will go from ', f_start, ' Hz to ', f_stop, ' Hz.')
    # Length in samples
    sweep_sample_length = T*fs
    # Round for actual number of samples
    np.round_(sweep_sample_length)
    # Convert float to int
    sweep_sample_length = int(sweep_sample_length)
    # Initialize time array
    t = np.arange(0, sweep_sample_length)/fs
    if sweep_type == 1:
        # Prepare linear frequency array
        f_sweep = np.linspace(f_start, f_stop, sweep_sample_length)
        # Compute linear sine sweep
        sweep = np.sin(2 * np.pi * f_sweep * t)
    elif sweep_type == 2:
        # Compute logarithmic sine sweep
        R = np.log(f_stop/f_start)
        sweep = np.sin((2*np.pi*f_start*T/R)*(np.exp(t*R/T)-1))
    else:
        # If choice invalid, say so
        print('Invalid choice, sorry! :\\')
    # Define as pyfar power signal
    sine_sweep_power = Signal(sweep, fs, signal_type='power')
    # Ask for desired filename (for wav)
    filename = input('Choose filename:\n'
                     '(If already existing, file will be overwritten!)\n')
    # Write the file
    pyfar.io.write_wav(sine_sweep_power, filename, overwrite=True)
    # Return signal
    return sine_sweep_power
