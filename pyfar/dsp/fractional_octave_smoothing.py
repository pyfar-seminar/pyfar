import numpy as np
from pyfar import Signal


class FractionalSmoothing:

    def __init__(
            self,
            sampling_rate,
            smoothing_width,
            win_type='rectangular'):
        """__init__ Initiate FractionalSmoothing object.

        :param sampling_rate: Sampling frequency of signal
        :type sampling_rate: float, int
        :param smoothing_width: Width of smoothing window relative to an octave
        :type smoothing_width: float, int
        :param win_type: Type of window applied to smooth,
         defaults to 'rectangular'
        :type win_type: str, optional
        :raises ValueError: Invalid data type of sampling_rate.
        :raises ValueError: Invalid data type of smoothing_width.
        :raises ValueError: Invalid window type.
        """
        if isinstance(sampling_rate,  (float, int)) is True:
            self._sampling_rate = sampling_rate
        else:
            raise ValueError("Invalid data type of sampling rate (int/float)")

        if isinstance(smoothing_width, (float, int)) is True:
            self._smoothing_width = smoothing_width
        else:
            raise ValueError("Invalid data type of window width (int/float)")

        self._VALID_WINDOW_TYPE = ["rectangular"]
        if (win_type in self._VALID_WIN_TYPE) is True:
            self._win_type = win_type
        else:
            raise ValueError("Not a valid window type ('rectangular')")

    def integration_limits(self):
        """integration_limits Computes integration limits for each frequency bin
        k. For each k two arrays are computed, one containing the upper, one
        containing the lower integration limits.
        The k arrays of upper and lower limits are stored in two arrays for
        upper and lower limits. Finally, the log2 on all elements is computed
        and the two arrays are combined to one. The array 'Phi' contains all
        integration limits and is stored in the FractionalSmoothing object.
        """
        # Lower and upper cutoff frequencies for each bin k:
        cutoff_low = self._disc_freq_bins*2**(-self._smoothing_width/2)
        cutoff_up = self._disc_freq_bins*2**(self._smoothing_width/2)
        # Range of frequency bins within upper and lower cutoff frequency:
        freq_bins = np.array([
                        np.arange(np.floor(k_c_low), np.ceil(k_c_up))
                        for k_c_low, k_c_up in zip(cutoff_low, cutoff_up)])
        # Replace first entry by lower cutoff frequency bin k'
        # and substract 0.5 from other entries:
        disc_freq_bins_low = np.array([
                        np.concatenate((np.array([k_L]), k[1:]-.5))
                        for k_L, k in zip(cutoff_low, freq_bins)])
        # Replace last entry by upper cutoff frequency bin k'
        # and add 0.5 to other entries:
        disc_freq_bins_up = np.array([
                        np.concatenate((k[:-1]+.5, np.array([k_U])))
                        for k, k_U in zip(freq_bins, cutoff_up)])
        # Divide by k and calc log:
        disc_freq_bins_low /= self._disc_freq_bins
        disc_freq_bins_up /= self._disc_freq_bins
        Phi_l = np.array([np.log2(i) for i in disc_freq_bins_low])
        Phi_u = np.array([np.log2(i) for i in disc_freq_bins_up])
        self._Phi = np.array([Phi_l, Phi_u])

    def calc_weights(self):
        if (self._win_type == "rectangular") is True:
            # TODO Computation: Upper - Lower / Delta
            self._weights = np.empty(
                [len(self._disc_freq_bins), len(self._disc_freq_bins)])

    # def apply(self, signal):
        