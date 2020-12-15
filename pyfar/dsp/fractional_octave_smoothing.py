import numpy as np
import cmath as cm
import copy
from pyfar import Signal


class FractionalSmoothing:

    def __init__(
            self,
            data,
            smoothing_width,
            win_type='rectangular'):
        """__init
        Initiate FractionalSmoothing object.

        :param  data:               Raw data of the signal in the
                                    frequency domain
        :type   data:               ndarray, double
        :param  sampling_rate:      Sampling frequency of signal
        :type   sampling_rate:      float, int
        :param  smoothing_width:    Width of smoothing window relative
                                    to an octave
        :type   smoothing_width:    float, int
        :param  win_type:           Type of window applied to smooth,
                                    defaults to 'rectangular'
        :type   win_type:           str, optional
        :raises TypeError:         Invalid data type of sampling_rate.
        :raises TypeError:         Invalid data type of smoothing_width.
        :raises TypeError:         Invalid window type.
        """
        if isinstance(data, np.ndarray) is True:
            if data.dtype == np.complex128:
                # Get number of freq bins from signal data
                self._n_bins = data.shape[-1]
                # Copy signal data
                self._data = np.atleast_2d(
                    np.asarray(data.copy(), dtype=np.complex))
            else:
                raise TypeError(
                    "ndarry must by of type: numpy.complex182.")
        else:
            raise TypeError(
                    "Invalid data type of input data (numpy.ndarray).")

        if isinstance(smoothing_width, (float, int)) is True:
            self._smoothing_width = smoothing_width
        else:
            raise TypeError("Invalid data type of window width (int/float).")

        self._VALID_WIN_TYPE = ["rectangular"]
        if (win_type in self._VALID_WIN_TYPE) is True:
            self._win_type = win_type
        else:
            raise TypeError("Not a valid window type ('rectangular').")

    def calc_integration_limits(self):
        """integration_limits
        Computes integration limits for each frequency bin
        k. For each k two arrays are computed, one containing the upper, one
        containing the lower integration limits.
        The k arrays of upper and lower limits are stored in two arrays for
        upper and lower limits. Finally, the log2 on all elements is computed
        and the two arrays are combined to one. The array 'Phi' contains all
        integration limits and is stored in the FractionalSmoothing object.
        """
        # Freq bin iterator:
        k_i = np.arange(self._n_bins)
        # Lower and upper cutoff frequencies bin for each bin k:
        k_cutoff_low = k_i*2**(-self._smoothing_width/2)
        k_cutoff_up = k_i*2**(self._smoothing_width/2)

        # Matrix with shape: 'number of freq bins' x 'number of freq bins' x 2:
        # Each row stores upper and lower limits for k. freq bin from 0 to N-1
        k_mat = np.array([
            lim_padder(low, up, self._n_bins) for low, up in
            zip(k_cutoff_low, k_cutoff_up)])

        # Divide by k:
        k_divider = np.array([k_i]*(k_mat.shape[2]*2)).T.reshape(k_mat.shape)
        k_mat /= k_divider

        # Apply log:
        self._limits = np.log2(k_mat)

        # Replace all -inf and nan by zero:
        self._limits = np.nan_to_num(self._limits, posinf=.0, neginf=.0)

    def calc_weights(self):
        """calc_weights
        Computes integration from lower to upper limits with
        choosen window type.
        'rectangular':
        """
        if (self._win_type == "rectangular") is True:
            # Computation: Upper - Lower / Smoothing_Width
            self._weights = (self._limits[:, 0] - self._limits[:, 1])
            self._weights /= self._smoothing_width
        # Set Weight for freq bin = 0 to 1 (no smoothing at 0 Hz)
        self._weights[0, 0] = 1

    def apply(self):
        """apply
        Apply weights to magnitude spectrum of signal and return new
        complex spectrum.

        :return: Complex spectrum
        :rtype: ndarray
        """
        magnitude = np.sum(self._weights*np.abs(self._data), axis=1)
        phase = np.angle(self._data)

        return cm.rect(magnitude, phase)


# Helper function to pad array of range to specified length:
def lim_padder(low, up, size):
    # Get rounded limits:
    # Round up for lower limit, round down for upper limits
    low_ceil = np.ceil(low)
    up_floor = np.floor(up)
    # Set base array beginning to lowest k'-.5 that is above limit "low"
    begin = low_ceil - 0.5 if (low_ceil - low) > .5 else low_ceil + 0.5
    # Set base array end to highest k'+5 that is below limit "up"
    end = up_floor + 1 if (up - up_floor) > .5 else up_floor
    # Create base array
    base = np.arange(begin, end)
    # Append actuall limits to base:
    # Push back for upper limit
    upper_limit = np.append(base, up)
    # Push front for lower limit
    lower_limit = np.append(low, base)
    # Pad arrays front:
    pad_front = int(upper_limit[0])
    upper_limit = np.pad(upper_limit, (pad_front, 0))
    lower_limit = np.pad(lower_limit, (pad_front, 0))
    # Pad array back:
    # if size is greater then array, add zeros
    if len(upper_limit) < size:
        upper_limit = np.pad(upper_limit, (0, size-len(upper_limit)))
        lower_limit = np.pad(lower_limit, (0, size-len(lower_limit)))
    # else cut away end of array, this is necessary when smoothing high freqs
    # at end of data vector, smoothing weights are computed for k in [0, N-1]
    # Alternative: pad data vector to fit size -> TODO
    else:
        upper_limit = upper_limit[:size]
        lower_limit = lower_limit[:size]
    # Return upper and lower limits of k-th freq bin in one array:
    return np.concatenate((
        upper_limit.reshape(1, -1),
        lower_limit.reshape(1, -1)))


def fractional_smooth(signal, smoothing_width, win_type='rectangular'):
    """fractional_smooth Method to smooth a given signal.
    Creates a object of class FractionalSmoothing to compute integration limits
    and smoothing weights. Finally applies the weights on the input data and
    returns smoothed signal.

    :param signal:  Input signal to be smoothed
    :type signal:   numpy.ndarray with np.complex128
    :param          smoothing_width:    Width of smoothing window relative
                                        to an octave
    :type           smoothing_width:    float, int
    :param          win_type:           Type of window applied to smooth,
                                        defaults to 'rectangular'
    :type           win_type:           str, optional
    :raises         TypeError:          Input data must be of type Signal.
    :return:        Smoothed Signal
    :rtype:         Signal
    """
    if isinstance(signal, Signal) is True:
        # Create object
        obj = FractionalSmoothing(
            data=signal.freq,
            smoothing_width=smoothing_width,
            win_type=win_type)
        # Compute limits:
        obj.calc_integration_limits()
        # Compute weights:
        obj.calc_weights()
        # Compute smoothed magnitude spectrum
        data = obj.apply()

        # Return smoothed signal
        return Signal(
            data,
            copy(signal.sampling_rate),
            copy(signal.n_samples),
            domain='freq')
    else:
        raise TypeError("Input data must be of type Signal.")
