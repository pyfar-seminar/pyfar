import numpy as np
from pyfar import Signal


class FractionalSmoothing:

    def __init__(
            self,
            data,
            smoothing_width,
            win_type='rectangular'):
        """__init__ Initiate FractionalSmoothing object.

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
        :raises ValueError:         Invalid data type of sampling_rate.
        :raises ValueError:         Invalid data type of smoothing_width.
        :raises ValueError:         Invalid window type.
        """
        if isinstance(data, np.ndarray) is True:
            if data.dtype == np.complex128:
                # Get freq bins from signal data
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

    @classmethod
    def from_signal(cls, signal, smoothing_width, win_type='rectangular'):
        if isinstance(signal, Signal) is True:
            return cls(
                data=signal.freq,
                smoothing_width=smoothing_width,
                win_type=win_type)
        else:
            raise TypeError("Input data must be of type Signal.")


    def integration_limits(self):
        """integration_limits Computes integration limits for each frequency bin
        k. For each k two arrays are computed, one containing the upper, one
        containing the lower integration limits.
        The k arrays of upper and lower limits are stored in two arrays for
        upper and lower limits. Finally, the log2 on all elements is computed
        and the two arrays are combined to one. The array 'Phi' contains all
        integration limits and is stored in the FractionalSmoothing object.
        """
        # Lower and upper cutoff frequencies bin for each bin k:
        k_cutoff_low = np.floor(
            self._disc_freq_bins*2**(
                -self._smoothing_width/2)).astype(np.int32)
        k_cutoff_up = np.ceil(
            self._disc_freq_bins*2**(self._smoothing_width/2)).astype(np.int32)
        # Combine to one element:
        k_cutoff = np.concatenate((
            k_cutoff_low.reshape(1, -1),
            k_cutoff_up.reshape(1, -1),
            self._disc_freq_bins.reshape(1, -1)
            )).T

        # Matrix with shape: 'number of freq bins' x 'max cutoff freq bin'
        # 'max cutoff freq bin' at k_cutoff[:,1][-1]
        # Each row stores upper or lower limits for k. freq bin
        k_mat_up = np.array(
            [lim_padder(k[0], k[1], k[:, 1][-1], .5) for k in k_cutoff])
        k_mat_down = np.array(
            [lim_padder(k[0], k[1], k[:, 1][-1], .5) for k in k_cutoff])
        # Combine matrices
        k_mat = np.array([k_mat_up, k_mat_down])

        # Replace first and last non-zero elements
        # by upper and lower limits of k:
        for idx, k_i in enumerate(k_cutoff):
            # Set upper limit [0] for k. freq bin [idx]
            # at position of last non-zero element k_i[1]-1:
            k_mat[0, idx, k_i[1]-1] = k_i[2]*2**(self._smoothing_width/2)
            # Set lower limit [1] for k. freq bin [idx]
            # at position of first non-zero element k_i[0]:
            k_mat[1, idx, k_i[0]] = k_i[2]*2**(-self._smoothing_width/2)

        # Divide by k:
        k_mat /= np.array(
            [np.array([k_cutoff.T[2]]*k_mat.shape[2]).T]*k_mat.shape[0])

        # Apply log:
        self._limits = np.log2(k_mat)

    def calc_weights(self):
        """calc_weights [summary] TODO
        """
        if (self._win_type == "rectangular") is True:
            # Computation: Upper - Lower / Smoothing_Width
            self._weights = (self._limits[0] - self._limits[1])
            self._weights /= self._smoothing_width

    # def apply(self, signal):


# Helper function to pad array of range to specified length:
def lim_padder(start, stop, length, add_val):
    np.pad(np.arange(start, stop) + add_val, (start, length))
