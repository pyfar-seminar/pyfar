# %%
from pyfar.dsp.fractional_octave_smoothing import FractionalSmoothing as fs
from pyfar.dsp import dsp as dsp
import numpy as np

# %% Difference when smoothing width is large:
# Weights with new version:
# Create smoothing object
n_bins = 100
smoothing_width = 10

# Eq. (17) - log integration limits
# phi_low and phi_high are largely identical - calculation could be
# made more efficient
k_max = int(np.ceil((n_bins-1)*2**(smoothing_width/2))+1)
k = np.atleast_2d(np.arange(k_max))
phi_low = np.log2((k.T - .5) / k[:, :n_bins])
phi_high = np.log2((k.T + .5) / k[:, :n_bins])

# Eq. (15) - window function at all phi
w_phi_low = (phi_low + smoothing_width/2) / smoothing_width
w_phi_low[phi_low < -smoothing_width/2] = 0
w_phi_low[phi_low > smoothing_width/2] = 1

w_phi_high = (phi_high + smoothing_width/2) / smoothing_width
w_phi_high[phi_high < -smoothing_width/2] = 0
w_phi_high[phi_high > smoothing_width/2] = 1

# Eq (16) - weights
weights = w_phi_high - w_phi_low
weights[0] = 0        # fix NaNs for k=0
weights[0, 0] = 1

# Transpose to fit old implementation:
weights = weights.T

np.sum(weights, axis=1)
# %% Weight with old version:
smoother = fs(n_bins, smoothing_width)
# Compute integration limits
limits = smoother._calc_integration_limits()
limits_low = (smoother._calc_integration_limits()[1]).toarray()
# Compute weights:
smoother._calc_weights_old()
weights = (smoother._weights).toarray()
np.sum(weights, axis=1)
# %% Compare lower limits:
print(f'Phi_min = -smoothing_width/2 = {-smoothing_width/2}')
print('k=1:')
print(f'New: {phi_low.T[1,:10]}')
print(f'Old: {limits_low[1,:10]}')
print('k=2:')
print(f'New: {phi_low.T[2,:10]}')
print(f'Old: {limits_low[2,:10]}')

# %% ---------------------------------------------------------------------------
# No difference when smoothing width is 2 or smaller:
#  Weights with new version:
# Create smoothing object
n_bins = 100
smoothing_width = 2

# Eq. (17) - log integration limits
# phi_low and phi_high are largely identical - calculation could be
# made more efficient
k_max = int(np.ceil((n_bins-1)*2**(smoothing_width/2))+1)
k = np.atleast_2d(np.arange(k_max))
phi_low = np.log2((k.T - .5) / k[:, :n_bins])
phi_high = np.log2((k.T + .5) / k[:, :n_bins])

# Eq. (15) - window function at all phi
w_phi_low = (phi_low + smoothing_width/2) / smoothing_width
w_phi_low[phi_low < -smoothing_width/2] = 0
w_phi_low[phi_low > smoothing_width/2] = 1

w_phi_high = (phi_high + smoothing_width/2) / smoothing_width
w_phi_high[phi_high < -smoothing_width/2] = 0
w_phi_high[phi_high > smoothing_width/2] = 1

# Eq (16) - weights
weights = w_phi_high - w_phi_low
weights[0] = 0        # fix NaNs for k=0
weights[0, 0] = 1

# Transpose to fit old implementation:
weights = weights.T

np.sum(weights, axis=1)
# %% Weight with old version:
smoother = fs(n_bins, smoothing_width)
# Compute integration limits
limits = smoother._calc_integration_limits()
limits_low = (smoother._calc_integration_limits()[1]).toarray()
# Compute weights:
smoother._calc_weights_old()
weights = (smoother._weights).toarray()
np.sum(weights, axis=1)
# %% Compare lower limits:
print(f'Phi_min = -smoothing_width/2 = {-smoothing_width/2}')
print('k=1:')
print(f'New: {phi_low.T[1,:10]}')
print(f'Old: {limits_low[1,:10]}')
print('----------------------------------------------------------------------')
print('k=2:')
print(f'New: {phi_low.T[2,:10]}')
print(f'Old: {limits_low[2,:10]}')
# %%
