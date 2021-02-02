import numpy as np
from pyfar import Signal
import scipy.sparse as sparse
# from pyfar import Signal


class FractionalSmoothing:
    """ Class of fractional smoothing object.
        Object contains data of a signal and a given smoothing width.

        References
        ----------
        .. [1] JO. G.. Tylka, BR. B.. Boren, and ED. Y.. Choueiri,
               "A Generalized Method for Fractional-Octave Smoothing of
               Transfer Functions that Preserves Log-Frequency Symmetry,"
               J. Audio Eng. Soc., vol. 65, no. 3, pp. 239-245, (2017 March.).
               doi: https://doi.org/10.17743/jaes.2016.0053
    """
    def __init__(
            self,
            n_bins,
            smoothing_width,
            phase_type=None):
        """
        Initiate FractionalSmoothing object.

        Parameters
        ----------
        n_bins : int
            Number of frequency bins of signal.
        smoothing_width : float, int
            Width of smoothing window relative to an octave.
            E.g.: smoothing_width=1/3 denotes third-octave smoothing.
        phase_type : str, default None.
            Phase type specifier: If `None` or `zero`, signal with zero phase
            is returned. `original` copies phase from input signal.
            TODO: other phase types.
        """
        if not isinstance(smoothing_width, (float, int)):
            raise TypeError("Invalid data type of window width (int/float).")
        if not isinstance(n_bins, int):
            raise TypeError("Invalid data type of number of bins (int).")
        self._VALID_PHASE_TYPE = [
            "Original", "Zero", "Minimum", "Linear"]

        # Set number of freq bins
        self._n_bins = n_bins
        # Set smoothing width:
        self._smoothing_width = smoothing_width
        # Calc weights:
        self.calc_weights()
        # Set update weight flag to False
        self._update_weigths = False
        # Set default phase type to Zero
        if not phase_type:
            self._phase_type = 'Zero'
        else:
            self._phase_type = phase_type

    def calc_integration_limits(self):
        """
        Compute integration limits for each frequency bin as in eq. (17) in
        _[1].

        For each frequency bin k two arrays are computed, one containing the
        upper, one containing the lower integration limits.
        The k arrays of upper and lower limits are stored in two arrays for
        upper and lower limits. Finally, the log2 on all elements is computed
        and the two arrays are combined to one.

        Returns
        -------
        ndarray
            Limits array of shape (n_bins, 2, max_smoothing_freq_bin).
        """
        if not self.n_bins:
            raise ValueError("Number of frequency bins not given.")
        if not self.smoothing_width:
            raise ValueError("Smoothing width not given.")
        # Freq bin iterator:
        k_i = np.arange(self.n_bins)
        # Lower and upper cutoff frequencies bin for each bin k:
        k_cutoff_low = k_i*2**(-self.smoothing_width/2)
        k_cutoff_up = k_i*2**(self.smoothing_width/2)
        # Matrix with shape: 'number of freq bins' x 'max. cutoff bin' x 2:
        # Each row stores upper and lower limits for k. freq bin from 0 to
        # max. cutoff freq:
        # k_cutoff_max = k_max*2**(self.smoothing_width/2) = k_cutoff_up[-1]
        size = int(np.ceil(k_cutoff_up[-1]) + 1)
        k_mat = np.array([
            self.lim_padder(low, up, size) for low, up in
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
        Calculates frequency dependent weights from limits as in eq. (16) in
        _[1].

        Each weight is computed from integrating over a rectangular window from
        lower to upper limit. A array of weights is stored in the object as
        scipy.sparse matrix. The weight for the 0 Hz bin is set to 1, thereby
        the magnitude at 0 Hz remains through the smoothing process.
        """
        # Get limits:
        limits = self.calc_integration_limits()
        # Computation: Upper - Lower / Smoothing_Width and store in array
        self._weights = limits[0] - limits[1]
        self._weights /= self.smoothing_width

    def apply(self, src):
        """
        Apply weights to magnitude spectrum of signal and return new signal
        with new complex spectrum as in eq. (1) in _[1].

        The weights matrix is repeated according to the number of overall
        channels. Further, the input data array is padded to fit the size of
        the weights matrix. By padding the input data array for each frequency
        bin by the mean value of the part, the input data array that is
        overlapped by the window, boundary effects at the end of the spectrum
        are minimized.
        After applying the weights, the padded part is removed again.
        The phase of the input data is treated according to the phase_type
        variable. See phase_type for more information.

        Parameters
        ----------
        src : signal
            Source signal.

        Returns
        -------
        ndarray
            Complex spectrum.
        """
        if not isinstance(src, Signal):
            raise TypeError("Invalid src input type (Signal).")
        if not src.n_bins == self.n_bins:
            raise ValueError("Input signal must have same number of frequencies \
                              bins as set in smoothing object. Set number of \
                              frequencies with obj.n_bins.")

        # Prepare source signal:
        # Copy flattened signal to buffer:
        signal_buffer = src.flatten()
        # Set buffer signal to frequency domain
        if not signal_buffer.domain == 'freq':
            signal_buffer.domain = 'freq'
        # Set FFT norm for input signal to "none":
        signal_buffer.fft_norm = 'none'
        # Get signal data:
        data_buffer = signal_buffer.freq.copy()

        # Check if weights need to be updated:
        if self._update_weigths:
            # Update weights
            self.calc_weights()
            # Reset flag
            self._update_weigths = False
        # Prepare weighting matrix:
        # Convert weights to array:
        weights = self._weights.todense()
        # Set Weight for freq bin = 0 to 1 (no smoothing at 0 Hz)
        weights[0, 0] = 1
        # Pad_width from difference of weights length and data length
        pad_width = weights.shape[1] - self.n_bins
        # Get size of signal that is used to calc mean value to pad:
        # Difference between data length and start of weighting window
        mean_size = self.n_bins - (weights != 0).argmax(axis=1)
        # Add new dimension for channels
        weights = np.expand_dims(weights, axis=0)
        # Expand weights matrix for src data
        weights = np.repeat(weights, data_buffer.shape[0], axis=0)

        # Prepare source signal data:
        # Pad data into array of weighting matrix shape
        # For each frequency bin k, data is padded according with
        # specified mean. The mean is computed from all values within
        # the range of the smoothing window of the particular frequency bin k
        magn_buffer = self.data_padder(np.abs(data_buffer),
                                       pad_width,
                                       mean_size)
        # Multiplication of weighting and data matrix along axis 2
        dst_magn = np.sum(weights*magn_buffer, axis=2)
        # Remove padded samples:
        dst_magn = dst_magn[:, :self.n_bins]

        # Phase handling:
        if self.phase_type == 'Original':
            # Copy phase from original data
            dst_phase = np.angle(data_buffer)
        if self.phase_type == 'Zero':
            # Copy phase from original data
            dst_phase = np.zeros_like(data_buffer)
        # TODO: other phase types.
        else:
            raise ValueError("Invalid Phase type given.")

        # Convert array in cartesian form:
        dst_data = dst_magn * np.exp(1j * dst_phase)
        # Return smoothed reshaped signal
        return Signal(dst_data, src.sampling_rate, src.n_samples, 'freq',
                      src.fft_norm, src.dtype, src.comment).reshape(src.cshape)

    @property
    def n_bins(self):
        """get_n_bins
        Return number of frequencies bins.

        Returns
        -------
        n_bins : int
            Number of frequency bins of signal.
        """
        return self._n_bins

    @n_bins.setter
    def n_bins(self, n_bins):
        """
        Sets number of frequencies bins.

        Parameters
        ----------
        n_bins : int
            Number of frequency bins of signal.

        Raises
        ------
        TypeError
            Invalid data type of n_bins (int).
        """
        if not isinstance(n_bins, int):
            raise TypeError("Invalid data type of number of bins (int).")
        # Get number of freq bins
        self.n_bins = n_bins
        # Update weight flag to True:
        self._update_weigths = True

    @property
    def smoothing_width(self):
        """
        Return smoothing width.

        Returns
        -------
        smoothing_width : float, int
            Width of smoothing window relative to an octave.
            E.g.: smoothing_width=1/3 denotes third-octave smoothing.
        """
        return self._smoothing_width

    @smoothing_width.setter
    def smoothing_width(self, smoothing_width):
        """
        Sets smoothing_width of the ovtace rectangular smoothing window.

        Parameters
        ----------
        smoothing_width : float, int
            Width of smoothing window relative to an octave.
            E.g.: smoothing_width=1/3 denotes third-octave smoothing.

        Raises
        ------
        TypeError
            Invalid data type of smoothing_width (int/float).
        """
        if not isinstance(smoothing_width, (float, int)):
            raise TypeError("Invalid data type of window width (int/float).")
        # Save smoothing width:
        self.smoothing_width = smoothing_width
        # Update weight flag to True:
        self._update_weigths = True

    @property
    def phase_type(self):
        """phase_type
        Specify how to treate phase of signal.

        'original'  Use phase of input signal.
        'zero'      Set phase of smoothed singal to zero.
        'minimum'   TODO
        'linear'    TODO

        Returns
        -------
        _VALID_PHASE_TYPE
            Phase type
        """
        return self._phase_type

    @phase_type.setter
    def phase_type(self, phase_type):
        """phase_type
        Set phase type.

        Parameters
        ----------
        phase_type : _VALID_PHASE_TYPE
            Phase type
        """
        if phase_type not in self._VALID_PHASE_TYPE:
            raise TypeError("Phase type must be one of the following: \
                            'original', 'zero', 'minimum', 'linear'.")
        # Save phase type:
        self.phase_type = phase_type

    @staticmethod
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
        # Pad array back to 'size' (length of largest smoothing window at
        # highest frequency bin k)
        upper_limit = np.pad(upper_limit, (0, size-len(upper_limit)))
        lower_limit = np.pad(lower_limit, (0, size-len(lower_limit)))
        # Return upper and lower limits of k-th freq bin in one array:
        return np.concatenate((
            upper_limit.reshape(1, -1),
            lower_limit.reshape(1, -1)))

    @staticmethod
    def data_padder(data, pad_width, mean_size):
        """
        Pads data array of shape (N, M) to data matrix of shape
        (N, M, M+pad_width). The output data contains M copies of the input
        data rows padded by 'pad_width' with the mean value of the last data
        samples. The number of samples used to compute the mean for each row is
        specified in the array 'mean_size'.

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
        padded = np.array([np.pad(data, ((0, 0), (0, pad_width)), 'mean',
                                  stat_length=(m)) for m in mean_size])
        # move channel axis to front and return
        return np.moveaxis(padded, 1, 0)
