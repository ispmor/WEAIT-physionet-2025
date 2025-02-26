import numpy as np
import pywt
import scipy
from pybaselines import Baseline

def sigma_est_dwt(detail_coeffs, distribution='Gaussian'):
    """Calculate the robust median estimator of the noise standard deviation.

    Parameters
    ----------
    detail_coeffs : ndarray
        The detail coefficients corresponding to the discrete wavelet
        transform of an image.
    distribution : str
        The underlying noise distribution.

    Returns
    -------
    sigma : float
        The estimated noise standard deviation (see section 4.2 of [1]_).

    References
    ----------
    .. [1] D. L. Donoho and I. M. Johnstone. "Ideal spatial adaptation
       by wavelet shrinkage." Biometrika 81.3 (1994): 425-455.
       :DOI:`10.1093/biomet/81.3.425`
    """
    # Consider regions with detail coefficients exactly zero to be masked out
    detail_coeffs = detail_coeffs[np.nonzero(detail_coeffs)]

    if distribution.lower() == 'gaussian':
        # 75th quantile of the underlying, symmetric noise distribution
        denom = scipy.stats.norm.ppf(0.75)
        sigma = np.median(np.abs(detail_coeffs)) / denom
    else:
        raise ValueError("Only Gaussian noise estimation is currently " "supported")
    return sigma



def _bayes_thresh(details, var):
    """BayesShrink threshold for a zero-mean details coeff array."""
    # Equivalent to:  dvar = np.var(details) for 0-mean details array
    dvar = np.mean(details * details)
    eps = np.finfo(details.dtype).eps
    thresh = var / np.sqrt(max(dvar - var, eps))
    return thresh



def wavelet_threshold(
    signal, coeffs,
    wavelet,
    threshold=None,
    sigma=None,
    mode='soft',
    wavelet_levels=None,
):

    dcoeffs = coeffs[1:]
    original_extent = tuple(slice(s) for s in signal.shape)


    if sigma is None:
        # Estimate the noise via the method in [2]_
        detail_coeffs = dcoeffs[-1]#['d' * signal.ndim]
        sigma = sigma_est_dwt(detail_coeffs, distribution='Gaussian')

    if threshold is None:
        var = sigma**2
        threshold = [
             _bayes_thresh(level, var) for level in dcoeffs
        ]

    if np.isscalar(threshold):
        # A single threshold for all coefficient arrays
        denoised_detail = [
            {
                key: pywt.threshold(level[key], value=threshold, mode=mode)
                for key in level
            }
            for level in dcoeffs
        ]
    else:
        # Dict of unique threshold coefficients for each detail coeff. array
        denoised_detail = [
            pywt.threshold(level, value=thresh, mode=mode) for thresh, level in zip(threshold, dcoeffs)
        ]
    denoised_coeffs = [coeffs[0]] + denoised_detail
    out = pywt.waverec(denoised_coeffs, wavelet)[original_extent]
    out = out.astype(signal.dtype)
    return out


def baseline_wandering_removal(signal, wavelet, level):
    N = len(signal[0])
    coeffs = pywt.wavedec(data=signal, wavelet=wavelet, level=level)
    reconstructed_approximates = [pywt.upcoef('a', coeffs[0][i], wavelet, level=level)[:N] for i in range(len(signal))]
    result = signal - reconstructed_approximates
    return result, coeffs




def remove_baseline_drift(recording):
    x=range(len(recording[0]))
    equalised = []
    params = []
    for y in recording:
        baseline_fitter = Baseline(x_data=x)
        bkg, params_local = baseline_fitter.snip(
            y, max_half_window=45, decreasing=True, smooth_half_window=4
        )
        bkg_to_equalise = bkg * -1
        equalised.append(y+bkg_to_equalise)
        params.append(params_local)
    return equalised, params
