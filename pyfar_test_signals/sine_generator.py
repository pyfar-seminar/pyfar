"""Generate colored noise.

Part of:
pyfar_test_signals
"""
import numpy as np
import pyfar
from pyfar import Signal    # managing audio signals


def sine():
    # Standard settings
    fs = 44100

    # 1 Sine was chosen
    # Confirmation
    print('You chose a sine.\n')
    # Ask to choose duration in seconds
    L = float(input('Choose duration in seconds:\n'))
    # Confirm duration
    print('The sine’s duration will be ', L, ' s.')
    # Ask to choose frequency
    f = float(input('Choose frequency:\n'))
    # Confirm frequency
    print('The sine’s frequency will be ', f, ' Hz.')
    # Compute sine with numpy
    sine = np.sin(2 * np.pi * f * np.arange(L*fs) / fs)
    # Create pyfar Signal object
    sine_power = Signal(sine, fs, signal_type='power')
    # Ask for desired filename (for wav)
    filename = input('Choose filename:\n'
                     '(If already existing, file will be overwritten!)\n')
    # Write file
    pyfar.io.write_wav(sine_power, filename, overwrite=True)
    # Return signal
    return sine_power
