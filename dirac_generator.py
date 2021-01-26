"""Generate a dirac delta impulse.

Part of:
pyfar_test_signals
"""
import numpy as np
import pyfar
from pyfar import Signal    # managing audio signals


def dirac():
    # Standard settings
    fs = 44100
    # 2 Dirac was chosen
    # Confirmation
    print('You chose a Dirac delta impulse.\n')
    # from pyfar examples (with adjustments):
    # Create a dirac signal with sampling rate fs
    # First sample (='head')
    head = 1
    # Rest of samples (='tail')
    tail = np.zeros(fs-1)
    # Combine head and tail
    dirac = np.concatenate((head, tail), axis=None)
    # Define as pyfar energy signal
    dirac_energy = Signal(dirac, fs, signal_type='energy')
    # show information
    # x_energy
    # Ask for desired filename (for wav)
    filename = input('Choose filename:\n'
                     '(If already existing, file will be overwritten!)\n')
    # Write the file
    pyfar.io.write_wav(dirac_energy, filename, overwrite=True)
    # Return signal
    return dirac_energy
