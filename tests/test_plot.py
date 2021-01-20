import matplotlib
import matplotlib.pyplot as plt
import matplotlib.testing as mpt
from matplotlib.testing.compare import compare_images
import pytest
import numpy as np
from unittest import mock
import os
import pyfar.plot as plot
from pyfar import Signal

# flag for creating new baseline plots (required if the plot look changed)
create_baseline = False

# path handling
base_path = os.path.join('tests', 'test_plot_data')
baseline_path = os.path.join(base_path, 'baseline')
output_path = os.path.join(base_path, 'output')

if not os.path.isdir(base_path):
    os.mkdir(base_path)
if not os.path.isdir(baseline_path):
    os.mkdir(baseline_path)
if not os.path.isdir(output_path):
    os.mkdir(output_path)

# figure parameters
f_width = 4.8
f_height = 4.8
f_dpi = 100


def test_line_plots(signal_mocks):
    """Test all line plots with default arguments and hold functionality."""

    function_list = [plot.line.time,
                     plot.line.freq,
                     plot.line.phase,
                     plot.line.group_delay,
                     plot.line.spectrogram,
                     plot.line.time_freq,
                     plot.line.freq_phase,
                     plot.line.freq_group_delay]

    for function in function_list:
        print(f"Testing: {function.__name__}")
        # file names
        filename = 'line_' + function.__name__ + '.png'
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)

        # plotting
        matplotlib.use('Agg')
        mpt.set_reproducibility_for_testing()
        plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi for testing
        function(signal_mocks[0])

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # testing
        compare_images(baseline, output, tol=10)

        # test hold functionality
        # file names
        filename = 'line_' + function.__name__ + '_hold.png'
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)

        # plotting
        function(signal_mocks[1])

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # close current figure
        plt.close()

        # testing
        compare_images(baseline, output, tol=10)


def test_line_phase_options(signal_mocks):
    """Test parameters that are unique to the phase plot."""

    parameter_list = [['line_phase_deg.png', True, False],
                      ['line_phase_unwrap.png', False, True],
                      ['line_phase_deg_unwrap.png', True, True]]

    for param in parameter_list:
        print(f"Testing: {param[0]}")
        # file names
        filename = param[0]
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)
        # plotting
        matplotlib.use('Agg')
        mpt.set_reproducibility_for_testing()
        plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi for testing
        plot.line.phase(signal_mocks[0], deg=param[1], unwrap=param[2])

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)
        # close current figure
        plt.close()

        # testing
        compare_images(baseline, output, tol=10)


def test_line_dB_option(signal_mocks):
    """Test all line plots that have a dB option."""

    function_list = [plot.line.time,
                     plot.line.freq]

    # test if dB option is working
    for function in function_list:
        for dB in [True, False]:
            print(f"Testing: {function.__name__} (dB={dB})")
            # file names
            filename = 'line_' + function.__name__ + '_dB_' + str(dB) + '.png'
            baseline = os.path.join(baseline_path, filename)
            output = os.path.join(output_path, filename)

            # plotting
            matplotlib.use('Agg')
            mpt.set_reproducibility_for_testing()
            plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi
            function(signal_mocks[0], dB=dB)

            # save baseline if it does not exist
            # make sure to visually check the baseline uppon creation
            if create_baseline:
                plt.savefig(baseline)
            # safe test image
            plt.savefig(output)

            # close current figure
            plt.close()

            # testing
            compare_images(baseline, output, tol=10)

    # test if log_prefix and log_reference are working
    for function in function_list:
        print(f"Testing: {function.__name__} (log parameters)")
        # file names
        filename = 'line_' + function.__name__ + '_logParams.png'
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)

        # plotting
        matplotlib.use('Agg')
        mpt.set_reproducibility_for_testing()
        plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi
        function(signal_mocks[0], log_prefix=10, log_reference=.5, dB=True)

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # close current figure
        plt.close()

        # testing
        compare_images(baseline, output, tol=10)


def test_line_xscale_option(signal_mocks):
    """Test all line plots that have an xscale option."""

    function_list = [plot.line.freq,
                     plot.line.phase,
                     plot.line.group_delay]

    # test if dB option is working
    for function in function_list:
        for xscale in ['log', 'linear']:
            print(f"Testing: {function.__name__} (xscale={xscale})")
            # file names
            filename = 'line_' + function.__name__ + '_xscale_' + xscale + \
                       '.png'
            baseline = os.path.join(baseline_path, filename)
            output = os.path.join(output_path, filename)

            # plotting
            matplotlib.use('Agg')
            mpt.set_reproducibility_for_testing()
            plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi
            function(signal_mocks[0], xscale=xscale)

            # save baseline if it does not exist
            # make sure to visually check the baseline uppon creation
            if create_baseline:
                plt.savefig(baseline)
            # safe test image
            plt.savefig(output)

            # close current figure
            plt.close()

            # testing
            compare_images(baseline, output, tol=10)


def test_time_unit(signal_mocks):
    """Test plottin with different units."""

    function_list = [plot.line.time,
                     plot.line.group_delay]

    for function in function_list:
        for unit in [None, 's', 'ms', 'mus', 'samples']:
            print(f"Testing: {function.__name__} (unit={unit})")
            # file names
            filename = f'line_{function.__name__}_unit_{str(unit)}.png'
            baseline = os.path.join(baseline_path, filename)
            output = os.path.join(output_path, filename)

            # plotting
            matplotlib.use('Agg')
            mpt.set_reproducibility_for_testing()
            plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi
            function(signal_mocks[1], unit=unit)

            # save baseline if it does not exist
            # make sure to visually check the baseline uppon creation
            if create_baseline:
                plt.savefig(baseline)
            # safe test image
            plt.savefig(output)

            # close current figure
            plt.close()

            # testing
            compare_images(baseline, output, tol=10)


def test_line_time_auto_unit():
    """Test automatically assigning the unit in group delay plots."""
    assert plot._line._time_auto_unit(0) == 's'
    assert plot._line._time_auto_unit(1e-4) == 'mus'
    assert plot._line._time_auto_unit(2e-2) == 'ms'
    assert plot._line._time_auto_unit(2) == 's'


def test_line_custom_subplots(signal_mocks):
    """
    Test custom subplots in row, column, and mixed layout including hold
    functionality.
    """

    # plot layouts to be tested
    plots = {
        'row': [plot.line.time, plot.line.freq],
        'col': [[plot.line.time], [plot.line.freq]],
        'mix': [[plot.line.time, plot.line.freq],
                [plot.line.phase, plot.line.group_delay]]
    }

    for p in plots:
        print(f"Testing: {p}")
        # file names
        filename = 'line_custom_subplots_' + p + '.png'
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)

        # plotting
        matplotlib.use('Agg')
        mpt.set_reproducibility_for_testing()
        plt.figure(1, (f_width, f_height), f_dpi)  # force size/dpi for testing
        plot.line.custom_subplots(signal_mocks[0], plots[p])

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # testing
        compare_images(baseline, output, tol=10)

        # test hold functionality
        # file names
        filename = 'line_custom_subplots_' + p + '_hold.png'
        baseline = os.path.join(baseline_path, filename)
        output = os.path.join(output_path, filename)

        # plotting
        plot.line.custom_subplots(signal_mocks[1], plots[p])

        # save baseline if it does not exist
        # make sure to visually check the baseline uppon creation
        if create_baseline:
            plt.savefig(baseline)
        # safe test image
        plt.savefig(output)

        # close current figure
        plt.close()

        # testing
        compare_images(baseline, output, tol=10)


def test_prepare_plot():
    # test without arguments
    plot._line._prepare_plot()

    # test with single axes object
    fig = plt.gcf()
    ax = plt.gca()
    plot._line._prepare_plot(ax)
    plt.close()

    # test with list of axes
    fig = plt.gcf()
    fig.subplots(2, 2)
    ax = fig.get_axes()
    plot._line._prepare_plot(ax)
    plt.close()

    # test with numpy array of axes
    fig = plt.gcf()
    ax = fig.subplots(2, 2)
    plot._line._prepare_plot(ax)
    plt.close()

    # test with list of axes and desired subplot layout
    fig = plt.gcf()
    fig.subplots(2, 2)
    ax = fig.get_axes()
    plot._line._prepare_plot(ax, (2, 2))
    plt.close()

    # test with numpy array of axes and desired subplot layout
    fig = plt.gcf()
    ax = fig.subplots(2, 2)
    plot._line._prepare_plot(ax, (2, 2))
    plt.close()

    # test without axes and with desired subplot layout
    plot._line._prepare_plot(None, (2, 2))
    plt.close()


@pytest.fixture
def signal_mocks():
    """ Generate two simple dirac signals for testing the hold functionality
    independently.

    Returns
    -------
    signal_1: Signal
        sine signal
    signal_2: Signal
        dirac signal

    """
    # signal parameters
    n_samples = int(2e3)
    n_bins = int(n_samples / 2) + 1
    sampling_rate = 40e3
    df = sampling_rate / n_samples
    times = np.arange(n_samples) / sampling_rate
    frequencies = np.arange(n_bins) * df

    # sine signal
    f = 10 * df
    time_sine = np.sin(2 * np.pi * f * times)
    freq_sine = np.zeros(n_bins)
    freq_sine[frequencies == f] = 1

    # impulse
    group_delay = 1000
    dphi = -group_delay / sampling_rate * df * 2 * np.pi
    time_impulse = np.zeros(n_samples)
    time_impulse[group_delay] = 1
    freq_impulse = 1 * np.exp(1j * np.arange(n_bins) * dphi)

    signal_1 = signal_mock(time_sine[np.newaxis, :], freq_sine[np.newaxis, :],
                           sampling_rate, times, frequencies,
                           'power', 'amplitude')

    signal_2 = signal_mock(time_impulse[np.newaxis, :],
                           freq_impulse[np.newaxis, :],
                           sampling_rate, times, frequencies,
                           'energy', 'unitary')

    return signal_1, signal_2


def signal_mock(time, freq, sampling_rate, times, frequencies,
                signal_type, fft_norm):
    """MagicMock for Signal including __getitem__ functionality. """

    # use MagicMock and side_effect to mock __getitem__
    # https://het.as.utexas.edu/HET/Software/mock/examples.html
    def getitem(slice):
        time = np.atleast_2d(signal.time[slice])
        freq = np.atleast_2d(signal.freq[slice])
        s_getitem = signal_mock(time, freq, signal.sampling_rate,
                                signal.times, signal.frequencies,
                                signal.signal_type, signal.fft_norm)
        return s_getitem

    # TODO: Implement __setitem__ if required in a test

    signal = mock.MagicMock(spec_set=Signal(time, sampling_rate))
    signal.time = time
    signal.sampling_rate = sampling_rate
    signal.times = times
    signal.n_samples = time.shape[-1]
    signal.freq = freq
    signal.frequencies = frequencies
    signal.n_bins = freq.shape[-1]
    signal.cshape = time.shape[:-1]
    signal.signal_type = signal_type
    signal.fft_norm = fft_norm
    signal.__getitem__.side_effect = getitem

    return signal
