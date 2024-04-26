from pathlib import Path

import nibabel as nb
import numpy as np
import pandas as pd
from nibabel import Nifti1Image
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.interfaces.fmriprep.tests._testing import create_tmp_filepath
from sklearn.preprocessing import scale


def _handle_non_steady(confounds):
    """Simulate non steady state correctly while increase the length.

    - The first row is non-steady state,
      replace it with the input from the second row.

    - Repeat X in length (axis = 0) 10 times to increase
      the degree of freedom for numerical stability.

    - Put non-steady state volume back at the first sample.
    """
    X = confounds.values
    non_steady = X[0, :]
    tmp = np.vstack((X[1, :], X[1:, :]))
    tmp = np.tile(tmp, (10, 1))
    return pd.DataFrame(np.vstack((non_steady, tmp[1:, :])), columns=confounds.columns)


def generate_data(demean):
    """Simulate an nifti image based on confound file \
    with some parts confounds and some parts noise."""
    file_nii, _ = create_tmp_filepath(Path(__file__).parent, copy_confounds=True)
    # set the size of the image matrix
    nx = 5
    ny = 5
    # the actual number of slices will actually be double of that
    # as we will stack slices with confounds on top of slices with noise
    nz = 2
    # Load a simple 6 parameters motion models as confounds
    # demean set to False just for simulating signal based on the original
    # state
    confounds, _ = load_confounds(
        file_nii, strategy=("motion",), motion="basic", demean=False
    )

    X = _handle_non_steady(confounds)
    X = X.values
    # the number of time points is based on the example confound file
    nt = X.shape[0]
    # initialize an empty 4D volume
    vol = np.zeros([nx, ny, 2 * nz, nt])
    vol_conf = np.zeros([nx, ny, 2 * nz])
    vol_rand = np.zeros([nx, ny, 2 * nz])

    # create random noise and a random mixture of confounds standardized
    # to zero mean and unit variance
    rng = np.random.default_rng(42)
    beta = rng.random((nx * ny * nz, X.shape[1]))
    tseries_rand = scale(rng.random((nx * ny * nz, nt)), axis=1)
    # create the confound mixture
    tseries_conf = scale(np.matmul(beta, X.transpose()), axis=1)

    # fill the first half of the 4D data with the random mixture
    vol[:, :, 0:nz, :] = tseries_conf.reshape(nx, ny, nz, nt)
    vol_conf[:, :, 0:nz] = 1

    # create random noise in the second half of the 4D data
    vol[:, :, range(nz, 2 * nz), :] = tseries_rand.reshape(nx, ny, nz, nt)
    vol_rand[:, :, range(nz, 2 * nz)] = 1

    # Shift the mean to non-zero
    vol = vol + 10

    # create an nifti image with the data, and corresponding mask
    img = Nifti1Image(vol, np.eye(4))
    mask_conf = Nifti1Image(vol_conf, np.eye(4))

    # generate the associated confounds for testing
    test_confounds, _ = load_confounds(
        file_nii, strategy=("motion",), motion="basic", demean=demean
    )
    # match how we extend the length to increase the degree of freedom
    test_confounds = _handle_non_steady(test_confounds)
    sample_mask = np.arange(test_confounds.shape[0])[1:]

    test_confounds.to_csv("confounds.tsv", index=False, sep="\t")
    nb.save(img, "input.nii.gz")
    nb.save(mask_conf, "mask.nii.gz")
    pd.DataFrame(sample_mask).to_csv("mask.tsv", index=False, sep="\t")
