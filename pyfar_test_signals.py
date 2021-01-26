#!/usr/bin/env python
"""Main module.
pyfar_test_signals

Copyright ©2021 Jonas Oertel

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
IN THE SOFTWARE.
"""

import pyfar
import pyfar.plot as plot
from sine_generator import sine
from dirac_generator import dirac
from noise_generator import noise
from sweep_generator import sweep

# Let user choose which signal they want to generate
selection = int(input('Which kind of signal would you like to generate?\n'
                      'Please choose the corresponding number:\n'
                      '1: Sine\n'
                      '2: Dirac delta impulse\n'
                      '3: Noise\n'
                      '4: Sine sweep\n'))

# Standard settings
fs = 44100

if selection == 1:
    # Call sine function
    sine_power = sine()
    # Plot sine function
    plot.line.freq(sine_power, style='dark')
elif selection == 2:
    # Call dirac function
    dirac_energy = dirac()
elif selection == 3:
    # Call noise function
    noise_power = noise()
elif selection == 4:
    # Call sweep function
    sweep_power = sweep()
else:
    # If choice invalid, say so
    print('Invalid choice! Sorry, try again :\\')
