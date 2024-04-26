import numpy as np
import pandas as pd
import pytest
from nilearn.maskers import NiftiMasker
from scipy.stats import pearsonr

from generate_data import generate_data

@pytest.mark.parametrize("standardize_signal", ["zscore", "psc"])
@pytest.mark.parametrize(
    "standardize_confounds,detrend",
    [(True, False), (False, True), (True, True)],
)
def test_nilearn_standardize_low(standardize_signal, standardize_confounds, detrend):
    """Test confounds removal with logical parameters for processing signal."""
    
    # demean is set to False to let signal.clean handle everything
    # generate_data(demean=False)

    df = pd.read_csv("mask.tsv", sep="\t")
    sample_mask = np.squeeze(df.values)

    # Extract time series with and without confounds.
    masker = NiftiMasker(
        mask_img="mask.nii.gz",
        standardize=standardize_signal,
        standardize_confounds=standardize_confounds,
        detrend=detrend,
    )
    tseries_raw = masker.fit_transform("input.nii.gz", sample_mask=sample_mask)
    tseries_clean = masker.fit_transform(
        "input.nii.gz", confounds="confounds.tsv", sample_mask=sample_mask
    )

    # the correlation before and after denoising should be very low
    # as most of the variance is removed by denoising
    corr = np.zeros(tseries_raw.shape[1])
    for ind in range(tseries_raw.shape[1]):
        corr[ind], _ = pearsonr(tseries_raw[:, ind], tseries_clean[:, ind])
    assert np.absolute(np.mean(corr)) < 0.2
