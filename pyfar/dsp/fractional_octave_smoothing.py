import numpy as np
import cmath as cm
import scipy.sparse as sparse
from pyfar import Signal


class FractionalSmoothing:
    """ Class of fractional smoothing object.
        Object contains data of a signal and a given smoothing width.
    """
    def __init__(
            self,
            n_bins,
            smoothing_width):
        """
        Initiate FractionalSmoothing object.

        Parameters
        ----------
        n_bins : int
            Number of frequency bins of signal.
        smoothing_width : float, int
            Width of smoothing window relative to an octave

        Raises
        ------
        TypeError
            Invalid data type of smoothing_width (int/float).
        TypeError
            Invalid data type of n_bins (int).
        """
        if not isinstance(smoothing_width, (float, int)):
            raise TypeError("Invalid data type of window width (int/float).")
        if not isinstance(n_bins, int):
            raise TypeError("Invalid data type of number of bins (int).")
        # Get number of freq bins
        self._n_bins = n_bins
        # Save smoothing width:
        self._smoothing_width = smoothing_width

    def calc_integration_limits(self):
        """
        Computes integration limits for each frequency bin
        k. For each k two arrays are computed, one containing the upper, one
        containing the lower integration limits.
        The k arrays of upper and lower limits are stored in two arrays for
        upper and lower limits. Finally, the log2 on all elements is computed
        and the two arrays are combined to one.

        Returns
        -------
        ndarray
            Limits array of shape (n_bins, 2, max_smoothing_freq_bin).
        """
        # Freq bin iterator:
        k_i = np.arange(self._n_bins)
        # Lower and upper cutoff frequencies bin for each bin k:
        k_cutoff_low = k_i*2**(-self._smoothing_width/2)
        k_cutoff_up = k_i*2**(self._smoothing_width/2)
        # Matrix with shape: 'number of freq bins' x 'max. cutoff bin' x 2:
        # Each row stores upper and lower limits for k. freq bin from 0 to
        # max. cutoff freq:
        # k_cutoff_max = k_max*2**(self._smoothing_width/2) = k_cutoff_up[-1]
        size = int(np.ceil(k_cutoff_up[-1]) + 1)
        k_mat = np.array([
            lim_padder(low, up, size) for low, up in
            zip(k_cutoff_low, k_cutoff_up)])
        # Divide by k:
        k_divider = np.array([k_i]*(k_mat.shape[2]*2)).T.reshape(k_mat.shape)
        k_mat /= k_divider
        # Apply log:
        limits = np.log2(k_mat)
        # Replace all -inf and nan by zero:
        limits = np.nan_to_num(limits, posinf=.0, neginf=.0)
        # Convert limits matrix to csr matrices
        return sparse.csr_matrix(limits[:, 0]), sparse.csr_matrix(limits[:, 1])

    def calc_weights(self):
        """calc_weights
        Calculates frequency dependent weights from limits. Each weight is
        computed from integrating over a rectangular window from lower to
        upper limit.
        A weights array is stored in the object. The weight for the 0 Hz bin
        is set to 1, thereby the value of 0 Hz remains through the smoothing
        process.
        """
        # Get limits:
        limits = self.calc_integration_limits()
        # Computation: Upper - Lower / Smoothing_Width and store in array
        self._weights = limits[0] - limits[1]
        self._weights /= self._smoothing_width

    def apply(self, data):
        """
        Apply weights to magnitude spectrum of signal and return new
        complex spectrum.
        The weights matrix is repeated according to the number of channels.
        Further, the data array is padded to fit the size of the weights
        matrix. This is done by padding the data array for each frequency bin
        by the mean value of the part of the data array that is overlapped by
        the window. This is done to avoid boundary effects at the end of the
        spectrum. After applying the weights to the padded data, the padded
        part is removed again.
        The phase of the data is copied to the smoothed data.

        Parameters
        ----------
        data : ndarray, double
            Raw data of the signal in the frequency domain

        Returns
        -------
        ndarray
            Complex spectrum.

        Raises
        ------
        TypeError
            Invalid data type of ndarray.
        TypeError
            Invalid data type of input data.
        ValueError
            Input data has to be same shape as set in n_bins in constructor.
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("Invalid data type of input data (numpy.ndarray).")
        if not data.dtype == np.complex128:
            raise TypeError("ndarry must by of type: numpy.complex182.")
        if not data.shape[-1] == self._n_bins:
            raise ValueError("Data must have shape of (#channel, {}).".format(
                              self._n_bins))
        # Get number of channels
        n_channels = data.shape[0]
        # Copy signal data
        data = np.atleast_2d(np.asarray(data.copy(), dtype=np.complex))
        # Convert weights to dense matrix:
        weights = self._weights.todense()
        # Set Weight for freq bin = 0 to 1 (no smoothing at 0 Hz)
        weights[0, 0] = 1
        # Pad_width from difference of weights length and data length
        pad_width = weights.shape[1] - self._n_bins
        # Get size of signal that is used to calc mean value to pad:
        # Difference between data length and start of weighting window
        mean_size = self._n_bins - (weights != 0).argmax(axis=1)
        # Add new dimension for channels
        weights = np.expand_dims(weights, axis=0)
        # Expand weights matrix for channels
        weights = np.repeat(weights, n_channels, axis=0)
        # Pad data into array of weighting matrix shape
        # For each frequency bin k, data is padded according with
        # specified mean. The mean is computed from all values within
        # the range of the smoothing window of the particular frequency bin k
        padded_data = data_padder(data, pad_width, mean_size)
        # Multiplication of weighting and data matrix along axis 2
        magnitude = np.sum(weights*padded_data, axis=2)
        # Remove padded samples:
        magnitude = magnitude[:, :self._n_bins]
        # Copy phase from original data
        phase = np.angle(data)
        # Return array in cartesian form:
        return polar2cartesian(magnitude, phase)


def lim_padder(low, up, size):
    """
    Returns a 2D array of shape (2, 'size') containing a range of
    k+.5 and k-.5 values with k increasing from 'low' to 'up' values.
    Arrays are padded by k-1 zeros befor and size - (K+1) after.
    First array: [0, ..., 0, k+.5, ..., K+.5, up, 0, ... ,0 ]
    Second array: [0, ..., 0, low, k-.5, ..., K-.5, 0, ..., 0]

    Parameters
    ----------
    low : float
        Lower value to start array
    up : float
        Upper value to end array
    size : int
        Total length of arrays

    Returns
    -------
    ndarray
        Double padded array
    """
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
    # Pad array back to 'size' (length of largest smoothing window at highest
    # frequency bin k)
    upper_limit = np.pad(upper_limit, (0, size-len(upper_limit)))
    lower_limit = np.pad(lower_limit, (0, size-len(lower_limit)))
    # Return upper and lower limits of k-th freq bin in one array:
    return np.concatenate((
        upper_limit.reshape(1, -1),
        lower_limit.reshape(1, -1)))


def data_padder(data, pad_width, mean_size):
    """
    Pads data array of shape (N, M) to data matrix of shape
    (N, M, M+pad_width). The output data contains M copies of the input data
    rows padded by 'pad_width' with the mean value of the last data samples.
    The number of samples used to compute the mean for each row is specified in
    the array 'mean_size'.

    Parameters
    ----------
    data : ndarray
        Array of shape (N, M) of the data.
        (N= Number of channels, M=Length of Signal)
    pad_width : int
        Padding length
    mean_size : ndarray
        Array with numbers of samples used to compute the mean.

    Returns
    -------
    ndarray
        Padded data matrix containing the data values and specified padding
        value for each frequency bin.

    Raises
    ------
    ValueError
        Data array must be 2D.
    """
    if not data.ndim == 2:
        raise ValueError('Data array must be 2D.')
    padded = np.array([np.pad(data, ((0, 0), (0, pad_width)),
                              'mean', stat_length=(m)) for m in mean_size])
    # move channel axis to front and return
    return np.moveaxis(padded, 1, 0)


def polar2cartesian(amplitude, phase):
    """
    Converts two arrays of amplitude and phase into one array
    of complex numbers in cartesian form.

    Parameters
    ----------
    amplitude : ndarray
        Array of amplitude values.
    phase : ndarray
        Array of phase values.

    Returns
    -------
    ndarray
        Array of same shape as input arrays with complex numbers.

    Raises
    ------
    ValueError
        Arrays must have same shape.
    """
    if not (amplitude.shape == phase.shape):
        raise ValueError("Arrays must have same shapes.")
    # Save input shape
    input_shape = amplitude.shape
    # Reshape for cmath rect to 1D arrays:
    reshaped_amplitude = amplitude.reshape(1, -1)[0]
    reshaped_phase = phase.reshape(1, -1)[0]
    # Apply cmath rect on elements:
    cartesian = np.array([cm.rect(a, p) for a, p in
                          zip(reshaped_amplitude, reshaped_phase)])
    return cartesian.reshape(input_shape)


def frac_smooth_signal(signal, smoothing_width):
    """
    Method to smooth a given signal.
    Takes the data of a given signal of shape (n, m), where n is number of
    channels and m length of the signal.
    Creates an object of class FractionalSmoothing to compute integration
    limits and smoothing weights. Finally applies the weights on the input data
    and returns smoothed signal.

    Parameters
    ----------
    signal : Signal
        Input signal to be smoothed
    smoothing_width : float, int
        Width of smoothing window relative to an octave

    Returns
    -------
    Signal
        Smoothed Signal

    Raises
    ------
    TypeError
        Input data must be of type Signal.
    """
    if not isinstance(signal, Signal):
        raise TypeError("Input data must be of type Signal.")
    # Create smoothing bject
    obj = FractionalSmoothing(signal.n_bins, smoothing_width)
    # Compute weights:
    obj.calc_weights()
    # Compute smoothed magnitude spectrum
    data = obj.apply(signal.freq)
    # Return smoothed signal
    return Signal(data, signal.sampling_rate, signal.n_samples, domain='freq')
