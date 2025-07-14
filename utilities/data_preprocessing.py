import logging
from scipy.stats import beta
import torch


logger = logging.getLogger(__name__)


def batch_preprocessing(batch):
    return multibranch_division(batch)


def multibranch_division(batch):
    x_raw, x_drift_removed,  y, rr_features, wavelet_features, recording_features = batch
    x_raw = torch.transpose(x_raw, 1, 2)
    x_drift_removed = torch.transpose(x_drift_removed, 1, 2)
    rr_features = torch.transpose(rr_features, 1, 2)
    wavelet_features = torch.transpose(wavelet_features, 1, 2)
    pre_pca = torch.hstack((x_drift_removed[:, ::2, :], wavelet_features, rr_features))


    pca_features = torch.pca_lowrank(pre_pca)
    pca_features = torch.hstack((pca_features[0].reshape(pca_features[0].shape[0], -1), pca_features[1], pca_features[2].reshape(pca_features[2].shape[0], -1)))
    pca_features = pca_features[:, :, None]

    #x_raw = torch.transpose(x_raw, 1, 2)
    #x_drift_removed = torch.transpose(x_drift_removed, 1, 2)
    #rr_features = torch.transpose(rr_features, 1, 2)
    #wavelet_features = torch.transpose(wavelet_features, 1, 2)
    #pca_features = torch.transpose(pca_features, 1, 2)


    alpha_input = x_raw
    beta_input = wavelet_features
    gamma_input = rr_features
    delta_input = pca_features
    epsilon_input = x_drift_removed

    logger.debug(f"Shape nf alpha_input: {alpha_input.shape}\nShape of beta_input: {beta_input.shape}\nGamma shape: {gamma_input.shape}\nDelta input shape: {delta_input.shape}\nEpsilon input shape: {epsilon_input.shape}")

    return alpha_input, beta_input, gamma_input, delta_input, epsilon_input,recording_features,  y
