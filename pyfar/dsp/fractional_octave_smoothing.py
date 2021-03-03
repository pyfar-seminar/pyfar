from enum import Enum
import numpy as np
from pyfar import Signal
import scipy.sparse as sparse
# from pyfar import Signal


class PhaseType(Enum):
    ZERO = 0
    ORIGINAL = 1
    MINIMUM = 2
    LINEAR = 3


class PaddingType(Enum):
    ZERO = 0
    EDGE = 1
    MEAN = 2


class FractionalSmoothing:
    """ Class of fractional smoothing object.
        Object contains data of a signal and a given smoothing width.

        References
        ----------
        .. [1] JO. G. Tylka, BR. B. Boren, and ED. Y. Choueiri,
               "A Generalized Method for Fractional-Octave Smoothing of
               Transfer Functions that Preserves Log-Frequency Symmetry,"
               J. Audio Eng. Soc., vol. 65, no. 3, pp. 239-245, (2017 March.).
               doi: https://doi.org/10.17743/jaes.2016.0053
    """

    def __init__(
            self,
            n_bins,
            smoothing_width,
            phase_type=PhaseType.ZERO,
            padding_type=PaddingType.MEAN):
        """
        Initiate FractionalSmoothing object.

        Parameters
        ----------
        n_bins : int
            Number of frequency bins of signal.
        smoothing_width : float, int
            Width of smoothing window relative to an octave.
            E.g.: smoothing_width=1/3 denotes third-octave smoothing.
        phase_type : PhaseType
            Phase type specifier: Default is PhaseType.ZERO: signal with zero
            phase is returned. PhaseType.ORIGINAL copies phase from input
            signal.
            TODO: other phase types.
        padding_tye : PaddingType
            Specify how to pad signal spectrum, when smoothing window is larger
            then greatest frequency.
        """
        if not isinstance(smoothing_width, (float, int)):
            raise TypeError("Invalid data type of window width (int/float).")
        if not isinstance(n_bins, int):
            raise TypeError("Invalid data type of number of bins (int).")
        if not isinstance(phase_type, PhaseType):
            raise TypeError("Invalid phase type.")
        if not isinstance(padding_type, PaddingType):
            raise TypeError("Invalid padding type.")

        # Set number of freq bins
        self._n_bins = n_bins
        # Set smoothing width:
        self._smoothing_width = smoothing_width
        # Set phase type:
        self._phase_type = phase_type
        # Set padding type:
        self._padding_type = padding_type
        # Calc weights:
        # self.calc_weights()
        # Set update weight flag to False
        self._update_weigths = True

    def _calc_integration_limits(self):
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
        # TODO: Warum +1?
        size = int(np.ceil(k_cutoff_up[-1])+1)
        k_mat = np.array([
            self._lim_padder(low, up, size) for low, up in
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

    def _calc_weights(self):
        """Calculates frequency dependent weights from limits as in eq. (16) in
        _[1].

        Each weight is computed from integrating over a rectangular window from
        lower to upper limit. A array of weights is stored in the object as
        scipy.sparse matrix. The weight for the 0 Hz bin is set to 1, thereby
        the magnitude at 0 Hz remains through the smoothing process.
        """
        # Get limits:
        limits = self._calc_integration_limits()
        # Computation: Upper - Lower / Smoothing_Width and store in array
        W = limits[0] - limits[1]
        W /= self.smoothing_width
        # Set Weight for freq bin = 0 to 1 (no smoothing at 0 Hz)
        W[0, 0] = 1
        self._weights = sparse.csr_matrix(W)

    def _calc_weights_new(self):
        """Calculates frequency dependent weights from limits as in eq. (16) in
        _[1].

        New version, does not need to compute limits first. 
        """
        # Eq. (17) - log integration limits
        # phi_low and phi_high are largely identical - calculation could be
        # made more efficient
        k_max = int(np.ceil((self._n_bins-1)*2**(self.smoothing_width/2)))
        k = np.atleast_2d(np.arange(k_max))
        phi_low = np.log2((k.T - .5) / k[:, :self._n_bins])
        phi_high = np.log2((k.T + .5) / k[:, :self._n_bins])

        # Eq. (15) - window function at all phi
        w_phi_low = (phi_low + self.smoothing_width/2) / self.smoothing_width
        w_phi_low[phi_low < -self.smoothing_width/2] = 0
        w_phi_low[phi_low > self.smoothing_width/2] = 1

        w_phi_high = (phi_high + self.smoothing_width/2) / self.smoothing_width
        w_phi_high[phi_high < -self.smoothing_width/2] = 0
        w_phi_high[phi_high > self.smoothing_width/2] = 1

        # Eq (16) - weights
        weights = w_phi_high - w_phi_low
        weights[0] = 0        # fix NaNs for k=0
        weights[0, 0] = 1

        # Transpose to fit old implementation:
        weights = weights.T
        # as sparse matrix
        self.weights = sparse.csr_matrix(weights)

    def apply_via_matrix(self, src):
        """Apply weights to magnitude spectrum of signal and return new signal
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
            raise ValueError("Input signal must have same number of "
                             "frequencies bins as set in smoothing object. "
                             "Set number of frequencies with obj.n_bins.")

        # Prepare source signal:
        # Copy flattened signal to buffer:
        src_copy = src.flatten()
        # Set buffer signal to frequency domain
        if src_copy.domain != 'freq':
            src_copy.domain = 'freq'
        # Set FFT norm for input signal to "none":
        src_copy.fft_norm = 'none'

        # ----------------------------------------------------------------------
        # Smoothing with matrix:

        # Check if weights need to be updated:
        if self._update_weigths:
            # Update weights
            self._calc_weights()
            # Reset flag
            self._update_weigths = False
        # Prepare weighting matrix:
        # Convert weights to array:
        weights = self._weights.todense()
        # Pad_width from difference of weights length and data length
        pad_width = weights.shape[1] - self.n_bins
        # Get size of signal that is used to calc mean value to pad:
        # Difference between data length and start of weighting window
        mean_size = self.n_bins - (weights != 0).argmax(axis=1)
        # Add new dimension for channels
        weights = np.expand_dims(weights, axis=0)
        # Expand weights matrix for src data
        weights = np.repeat(weights, src_copy.cshape, axis=0)
        # Prepare source signal data:
        src_magn_padded = self._data_padder(np.abs(src_copy.freq), pad_width,
                                            mean_size, self.padding_type)
        # Multiplication of weighting and data matrix along axis 2
        dst_magn = np.sum(weights*src_magn_padded, axis=2)

        # ----------------------------------------------------------------------

        # Remove padded samples:
        dst_magn = dst_magn[:, :self.n_bins]

        # Phase handling:
        if self.phase_type == PhaseType.ORIGINAL:
            # Copy phase from original data
            dst_phase = np.angle(src_copy.freq)
        elif self.phase_type == PhaseType.ZERO:
            # Copy phase from original data
            dst_phase = np.zeros_like(src_copy.freq)
        # TODO: other phase types.
        elif self.phase_type == PhaseType.MINIMUM:
            raise ValueError("PhaseType.MINIMUM is not implemented.")
        elif self.phase_type == PhaseType.LINEAR:
            raise ValueError("PhaseType.LINEAR is not implemented.")
        else:
            raise ValueError("Invalid phase type.")

        # Convert array in cartesian form:
        dst_data = dst_magn * np.exp(1j * dst_phase)
        # Create return object:
        dst = Signal(dst_data, src.sampling_rate, src.n_samples, 'freq',
                     'none', src.dtype, src.comment).reshape(src.cshape)
        # Set fft norm as in src:
        dst.fft_norm = src.fft_norm
        # Return smoothed reshaped signal
        return dst.reshape(src.cshape)

    def apply(self, src):
        """New method to smooth signal using a loop over the spectrum computing
        the weights at each frequency.

        Parameters
        ----------
        src : Signal
            Input signal that is to be smoothed.

        Returns
        -------
        Signal
            Smoothed signal of same shape and same metadata.
        """
        if not isinstance(src, Signal):
            raise TypeError("Invalid src input type (Signal).")
        if not src.n_bins == self.n_bins:
            raise ValueError("Input signal must have same number of "
                             "frequencies bins as set in smoothing object. "
                             "Set number of frequencies with obj.n_bins.")

        # Prepare source signal:
        # Copy flattened signal to buffer:
        src_copy = src.flatten()
        # Set buffer signal to frequency domain
        if src_copy.domain != 'freq':
            src_copy.domain = 'freq'
        # Set FFT norm for input signal to "none":
        src_copy.fft_norm = 'none'

        # ----------------------- SMOOTHING BY LOOP ----------------------------
        # Empty dst magnitude array:
        dst_magn = np.empty_like(src_copy.freq)
        # Loop over frequencies, skip k=0
        dst_magn[:, 0] = np.abs(src_copy.freq)[:, 0]
        # Max smoothing bin:
        k_max = int(np.ceil(src.n_bins*2**(self.smoothing_width/2)))
        # Padding length:
        pad_length = k_max - src.n_bins

        # Padding type:
        if self.padding_type == PaddingType.EDGE:
            src_magn_padded = np.pad(np.abs(src_copy.freq),
                                     ((0, 0), (0, pad_length)), 'edge')
        elif self.padding_type == PaddingType.ZERO:
            src_magn_padded = np.pad(np.abs(src_copy.freq),
                                     ((0, 0), (0, pad_length)))
        elif self.padding_type == PaddingType.MEAN:
            # Padding is done inside loop
            src_magn = np.abs(src_copy.freq)
        else:
            raise ValueError("Invalid padding type.")

        for k in range(1, src.n_bins):
            # ki array lenght +1 to fit size after np.ediff1d
            # ki = k'-0.5 = [-.5, .5, 1.5, ... k_max-.5]
            ki = np.arange(k_max+1)-.5
            # Min k' = k*2^(-win/2)
            ki_min = k*2**(-self.smoothing_width/2)
            # Replace all entries smaller then minimum by minimum
            ki[ki < ki_min] = ki_min
            # Max k' = k*2^(win/2)
            ki_max = k*2**(self.smoothing_width/2)
            # Replace all entries greater then maximum by maximum
            ki[ki > ki_max] = ki_max
            # Solving integral eq. (16)
            # (log2(ki+.5) - log2(ki-.5)) / smoothing_width
            W = np.ediff1d(np.log2(ki))/self.smoothing_width
            # Mean padding:
            if self.padding_type == PaddingType.MEAN:
                mean_length = src.n_bins - (W != 0).argmax(axis=0)
                src_magn_padded = np.pad(src_magn,
                                         ((0, 0), (0, pad_length)),
                                         'mean',
                                         stat_length=(mean_length))
            # Apply weights of freq bin k:
            dst_magn[:, k] = np.sum(W*src_magn_padded, axis=1)
        # ----------------------------------------------------------------------

        # Remove padded samples:
        dst_magn = dst_magn[:, :self.n_bins]

        # Phase handling:
        if self.phase_type == PhaseType.ORIGINAL:
            # Copy phase from original data
            dst_phase = np.angle(src_copy.freq)
        elif self.phase_type == PhaseType.ZERO:
            # Copy phase from original data
            dst_phase = np.zeros_like(src_copy.freq)
        # TODO: other phase types.
        elif self.phase_type == PhaseType.MINIMUM:
            raise ValueError("PhaseType.MINIMUM is not implemented.")
        elif self.phase_type == PhaseType.LINEAR:
            raise ValueError("PhaseType.LINEAR is not implemented.")
        else:
            raise ValueError("Invalid phase type.")

        # Convert array in cartesian form:
        dst_data = dst_magn * np.exp(1j * dst_phase)
        # Create return object:
        dst = Signal(dst_data, src.sampling_rate, src.n_samples, 'freq',
                     'none', src.dtype, src.comment).reshape(src.cshape)
        # Set fft norm as in src:
        dst.fft_norm = src.fft_norm
        # Return smoothed reshaped signal
        return dst.reshape(src.cshape)

    @property
    def n_bins(self):
        """Return number of frequencies bins.

        Returns
        -------
        n_bins : int
            Number of frequency bins of signal.
        """
        return self._n_bins

    @n_bins.setter
    def n_bins(self, n_bins):
        """Sets number of frequencies bins.

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
        if self._n_bins != n_bins:
            # Get number of freq bins
            self._n_bins = n_bins
            # Update weight flag to True:
            self._update_weigths = True

    @property
    def smoothing_width(self):
        """Return smoothing width.

        Returns
        -------
        smoothing_width : float, int
            Width of smoothing window relative to an octave.
            E.g.: smoothing_width=1/3 denotes third-octave smoothing.
        """
        return self._smoothing_width

    @smoothing_width.setter
    def smoothing_width(self, smoothing_width):
        """Sets smoothing_width of the ovtace rectangular smoothing window.

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
        if self._smoothing_width != smoothing_width:
            # Save smoothing width:
            self._smoothing_width = smoothing_width
            # Update weight flag to True:
            self._update_weigths = True

    @property
    def phase_type(self):
        """Specify how to treate phase of signal.

        ORIGINAL  Use phase of input signal.
        ZERO      Set phase of smoothed singal to zero.
        MINIMUM   TODO
        LINEAR    TODO

        Returns
        -------
        PhaseType
            Phase type
        """
        return self._phase_type

    @phase_type.setter
    def phase_type(self, phase_type):
        """Set phase type.

        Parameters
        ----------
        phase_type : _VALID_PHASE_TYPE
            Phase type
        """
        if not isinstance(phase_type, PhaseType):
            raise TypeError("Invalid phase type.")
        # Save phase type:
        self._phase_type = phase_type

    @property
    def padding_type(self):
        """Specfiy how to pad signal at the end of the sprectrum.

        The signal's spectrum has to be padded to fit the size of the smoothing
        window at high frequencies.
        The following three padding types are available:

        ZERO    Pads signal with zeros.
        EDGE    Pads signal with edge value (value at highes frequency bin).
        MEAN    Pads signal for each frequency by the mean value of the signal
                that is overlapped by the smoothing window.

        Returns
        -------
        PaddingType
            Padding type.
        """
        return self._padding_type

    @padding_type.setter
    def padding_type(self, padding_type):
        """Set padding type.

        Parameters
        ----------
        padding_type : PaddingType
            Padding type.

        """
        if not isinstance(padding_type, PaddingType):
            raise TypeError("Invalid padding type.")
        # Save padding type:
        self._padding_type = padding_type

    @staticmethod
    def _lim_padder(low, up, size):
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
    def _data_padder(data, pad_width, mean_size, padding_type):
        """
        Pads data array of shape (N, M) to data matrix of shape
        (N, N, M+pad_width). The output data contains M copies of the input
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
        padding_type : PaddingType

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
        if padding_type == PaddingType.MEAN:
            # Pad data into array of weighting matrix shape
            # For each frequency bin k, data is padded according with
            # specified mean. The mean is computed from all values within the
            # range of the smoothing window of the particular frequency bin k
            padded = np.array([np.pad(data, ((0, 0), (0, pad_width)), 'mean',
                                      stat_length=(m)) for m in mean_size])
        elif padding_type == PaddingType.EDGE:
            padded = np.array([np.pad(np.abs(data),
                                      ((0, 0), (0, pad_width)), 'edge')
                               for m in mean_size])
        elif padding_type == PaddingType.ZERO:
            padded = np.array([np.pad(np.abs(data),
                                      ((0, 0), (0, pad_width)))
                               for m in mean_size])
        else:
            raise ValueError(
                'PaddingType.MEAN not implemented for loop method.')

        # move channel axis to front and return
        return np.moveaxis(padded, 1, 0)
