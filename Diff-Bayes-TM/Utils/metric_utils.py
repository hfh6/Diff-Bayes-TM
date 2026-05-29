import numpy as np
import scipy.stats


def NMAE(reals, fakes, masks=None):
    if masks is None:
        masks = np.ones_like(reals).astype(np.bool_)
    nmae = np.abs(fakes[masks] - reals[masks]).sum() / np.abs(reals[masks]).sum()
    return nmae


def NRMSE(reals, fakes, masks=None):
    if masks is None:
        masks = np.ones_like(reals).astype(np.bool_)
    nrmse = np.sqrt(np.square(fakes[masks] - reals[masks]).sum()) / np.sqrt(np.square(reals[masks]).sum())
    return nrmse


def get_result(results):
   mean = np.mean(results)
   sigma = scipy.stats.sem(results)
   sigma = sigma * scipy.stats.t.ppf((1 + 0.95) / 2., 5-1)
   return mean, sigma
