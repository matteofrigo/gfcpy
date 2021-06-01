#!/usr/bin/env python
"""
Author: Matteo Frigo - matteo.frigo@inria.fr
"""
import argparse
import logging
import os

import nibabel as nib
import numpy as np


def check_can_write_file(fpath: str, force: bool = False):
    """
    Check if a file can be written.

    The function checks if the file already exists, the user has the permission
    to write it, overwriting can be forced and, if the file does not exist, if
    the parent directory exists and is writable.

    Args:
        fpath: str
            path of the file to be checked.
        force: bool
            True if the file can be overwritten, False otherwise.

    Raises:
        FileExistsError : if the file exists and can not be overwritten.
        PermissionError :  if the file esists and the user does not have the
            permission to write it.
        PermissionError : if the file does not exist, the parent directory
            exists and the user does not have the permission to write a file in
            it.
        FileNotFoundError : if file does not exist and the parent directory
            does not exist.
    """
    if os.path.exists(fpath) and os.path.isfile(fpath):
        if os.access(fpath, os.W_OK):
            if force:
                return
            else:
                raise FileExistsError(f'Specify `--force` to overwrite '
                                      f'{fpath}')
        else:
            # Tests for this case seem to be platform-dependent, hence have
            # been removed from the testing suite.
            raise PermissionError(f'User does not have permission to write '
                                  f'{fpath}')
    else:
        d = os.path.dirname(os.path.abspath(fpath))
        if os.path.exists(d):
            if os.access(d, os.W_OK):
                return
            else:
                raise PermissionError(f'User does not have permission to '
                                      f'write file in directory {d}')
        else:
            raise FileNotFoundError(f'Directory does not exist: {d}')


def extract_time_series(volume: np.ndarray, mask: np.ndarray, roi=None) -> \
        np.ndarray:
    """
    Extract the time series of each region or voxel.

    Args:
        volume:
            4-dimentional numpy array where the first three indices are the
            locations.
        mask:
            3-dimentional numpy array with the mask to be applied to the volume.
        roi:
            TODO

    Returns:
        2-dimentional numpy array with one time series per row. The number of
        rows is equal either to the number of voxels in the mask or to the
        number of ROIs.
    """
    if roi is None:
        ts = np.zeros((np.count_nonzero(mask), volume.shape[-1]))
        for count, (i, j, k) in enumerate(zip(*np.where(mask))):
            ts[count, :] = volume[i, j, k, :]
        logging.info(f'Extracted {ts.shape[0]} time series from voxels')
    else:
        labels = np.unique(roi)
        if 0 in labels:
            labels = labels[labels != 0]
        ts = np.zeros((len(labels), volume.shape[-1]))
        for count, l in enumerate(labels):
            if l == 0:
                continue
            series = np.zeros((np.count_nonzero(roi == l), volume.shape[-1]))
            for idx, (i, j, k) in enumerate(zip(*np.where(roi == l))):
                series[idx] = volume[i, j, k, :]
            ts[count, :] = np.mean(series, axis=0)
        logging.info(f'Extracted {ts.shape[0]} time series from ROIs.')
    return ts.astype(np.float32)


def pearson_correlation(ts: np.ndarray) -> np.ndarray:
    """
    Pearson's correlation coefficient.

    This function computes the Pearson's correlation coefficient between each
    pair of the given time series.

    Args:
        ts: np.ndarray
            2-dimensional array where each row contains a time series.

    Returns:
        correlation matrix between each pair of time series.
    """
    if not isinstance(ts, np.ndarray):
        raise TypeError('Input must be a numpy array.')

    if ts.ndim != 2:
        raise ValueError('Input array must be 2-dimensional.')

    n = ts.shape[1]  # number of samples in each time series
    e_xy = ts @ ts.T / n  # E[XY]
    e_x = np.mean(ts, axis=1)  # E[X]
    e_squaredx = np.mean(ts * ts, axis=1)  # E[X**2]

    num = e_xy - np.outer(e_x, e_x)  # E[XY] - E[X] - E[Y]
    den = np.sqrt(e_squaredx - e_x * e_x)  # sqrt(E[X**2] - E[X]**2)
    return num / np.outer(den, den)


def extract_absolute_gfc(c: np.ndarray) -> np.ndarray:
    """
    Extract global functional connectivity from connectivity matrix.

    The GFC is compute as the mean FC of each time series with all the others.

    Args:
        c:
            2-dimensional numpy array where each entry is the correlation
            between the corresponding pair of time series (voxel- or ROI-based).

    Returns:
        1-dimensional numpy array with the GFCs.
    """
    return (np.sum(np.abs(c), axis=1) - 1) / c.shape[1]


def extract_positive_gfc(c: np.ndarray) -> np.ndarray:
    """
    Extract global functional connectivity from connectivity matrix.

    The GFC is compute as the mean FC of each time series with all the others.

    Args:
        c:
            2-dimensional numpy array where each entry is the correlation
            between the corresponding pair of time series (voxel- or ROI-based).

    Returns:
        1-dimensional numpy array with the GFCs.
    """
    mask = (c > 0).astype(np.int)
    return (np.sum(c * mask, axis=1) - 1) / (c.shape[1] - 1)


def extract_negative_gfc(c: np.ndarray) -> np.ndarray:
    """
    Extract global functional connectivity from connectivity matrix.

    The GFC is compute as the mean FC of each time series with all the others.

    Args:
        c:
            2-dimensional numpy array where each entry is the correlation
            between the corresponding pair of time series (voxel- or ROI-based).

    Returns:
        1-dimensional numpy array with the GFCs.
    """
    mask = (c < 0).astype(np.int)
    return np.sum(c * mask, axis=1) / (c.shape[1] - 1)


def project_gfc_onto_voxel(gfc: np.ndarray, mask: np.ndarray,
                           volume: np.ndarray):
    """
    Project the input GFC values onto each voxel in a mask.

    Args:
        gfc:
            1-dimensional numpy array with the values of the GFCs.
        mask:
            3-dimensional array of booleans where the True values are the voxels
            on which the result will be projected.
        volume:
            3-dimensional array where the projection is written.
    """
    logging.info('Projecting GFC onto voxels')
    for value, i, j, k in zip(gfc, *np.where(mask)):
        volume[i, j, k] = value


def project_gfc_onto_roi(gfc: np.ndarray, roi: np.ndarray, volume: np.ndarray):
    """
    Project the input GFC values onto each voxel in a mask.

    Args:
        gfc:
            1-dimensional numpy array with the values of the GFCs.
        mask:
            3-dimensional array of booleans where the True values are the voxels
            on which the result will be projected.
        volume:
            3-dimensional array where the projection is written.
    """
    labels = np.unique(roi)
    if 0 in labels:
        labels = labels[labels != 0]
    if labels.size != np.asarray(gfc).size:
        raise ValueError('Size of labels and gfc mismatch.')

    for l, c in zip(labels, gfc):
        if l == 0:
            continue
        volume[roi == l] = c


def compute_z_score(c: np.ndarray) -> np.ndarray:
    """
    Compute the Z transform of the Pearson's correlation coefficient.

    z = arctanh(c)

    Args:
        c:
            Numpy array with the correlation coefficients.

    Returns:
        Numpy array with the same shape as the input and the values are the Z
        transform of the input.
    """
    return np.arctanh(c)


def get_parsed_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'in_fmri',
        type=str,
        help='The path to the fMRI data to be used as a 4D volume. '
             'Time must be the fourth dimension. '
             'Must be in NiBabel-readable format. '
    )
    parser.add_argument(
        'out_gfc',
        type=str,
        help='Prefix of the location where the volumes with the absolute, '
             'positive and negative GFC of each voxel/ROI will be saved in '
             'Nifti1 format. '
             'The space will be copied from the input fMRI. '
             'The saved files are: `<prefix>abs.nii.gz`, `<prefix>neg.nii.gz`'
             ' and `<prefix>pos.nii.gz`.'
    )

    parser.add_argument(
        '--in_mask',
        type=str,
        help='Mask to be used for selecting the portion of volume to be '
             'studied. '
             'Must be in NiBabel-readable format. '
    )

    parser.add_argument(
        '--out_txt',
        action='store_true',
        help='If set, saves the GFC of each ROI in a text file. '
             'Each row of the file contains the index of the ROI, the '
             'corresponding GFC and the Z score (if `--out_z` is set). '
             'The argument is ignored if `--roi` is not specified. '
             'The saved files are: `<prefix>abs.txt`, '
             '`<prefix>neg.txt` and `<prefix>pos.txt`.'
    )

    parser.add_argument(
        '--out_z',
        action='store_true',
        help='If set, saves the Z-score of the absolute, positive and negative '
             'GFC of each voxel/ROI will be saved in Nifti1 format. '
             'The space will be copied from the input fMRI. '
             'The saved files are: `<prefix>abs_z.nii.gz`, '
             '`<prefix>neg_z.nii.gz` and `<prefix>pos_z.nii.gz`.'

    )

    parser.add_argument(
        '--roi',
        type=str,
        help='Path to the ROI atlas to be used for the estimation of the GFC. '
             'The zero label is always ignored. '
             'Must be in NiBabel-readable format.'
    )

    parser.add_argument(
        '--force',
        action='store_true',
        help="Overwrite existing files."
    )

    verb = parser.add_mutually_exclusive_group()
    verb.add_argument(
        '--quiet',
        action='store_true',
        help='Do not display messages.'
    )
    verb.add_argument(
        '--warn',
        action='store_true',
        help='Display warning messages.'
    )
    verb.add_argument(
        '--info',
        action='store_true',
        help='Display information messages.'
    )
    verb.add_argument(
        '--debug',
        action='store_true',
        help='Display debug messages.'
    )

    return parser.parse_args()


def main(in_fmri=None, out_gfc=None, in_mask=None, roi=None, out_z=False,
         out_txt=False, force=False, *args, **kwargs):
    if not force:
        suffix = ['abs.nii.gz', 'neg.nii.gz', 'pos.nii.gz']
        if out_z:
            suffix.extend(['abs_z.nii.gz', 'neg_z.nii.gz', 'pos_z.nii.gz'])
        for f in [out_gfc + s for s in suffix]:
            check_can_write_file(f, force)

    logging.debug(f'Loading fmri data from {in_fmri}')
    fmri = nib.load(in_fmri)
    volume = fmri.get_fdata().astype(np.float32)
    affine = fmri.affine.copy()
    logging.info(f'Loaded fmri data of shape {volume.shape}')

    mask = np.sum(np.abs(volume), axis=3) > 0
    if in_mask is not None:
        volume_mask = nib.load(in_mask)

        pixdim_mask = volume_mask.header.get('pixdim')[1:4]
        pixdim_fmri = fmri.header.get('pixdim')[1:4]
        if not np.all(pixdim_fmri == pixdim_mask):
            raise ValueError('fMRI and mask have different voxel sizes.')

        shape_mask = volume_mask.shape
        shape_fmri = fmri.shape[:-1]
        if not np.all(shape_fmri == shape_mask):
            raise ValueError('fMRI and mask are not in the same space.')

        mask = np.logical_and(mask, volume_mask.get_fdata() > 0)
        logging.debug(f'Loaded mask from {in_mask}')

    if roi is not None:
        logging.debug(f'Using ROIs from {roi}')
        volume_roi = nib.load(roi)
        pixdim_volume = volume_roi.header.get('pixdim')[1:4]
        pixdim_fmri = fmri.header.get('pixdim')[1:4]
        if not np.all(pixdim_fmri == pixdim_volume):
            raise ValueError('fMRI and ROI volume have different voxel sizes.')

        shape_roi = volume_roi.shape
        shape_fmri = fmri.shape[:-1]
        if not np.all(shape_fmri == shape_roi):
            raise ValueError('fMRI and ROI volume are not in the same space.')

        roi = volume_roi.get_fdata()

    ts = extract_time_series(volume, mask, roi=roi)
    correlations = pearson_correlation(ts)

    gfc_type = {'abs': extract_absolute_gfc, 'pos': extract_positive_gfc,
                'neg': extract_negative_gfc}

    for k, gfc_fun in gfc_type.items():
        gfc = gfc_fun(correlations)

        volume = np.zeros_like(mask, dtype=np.float32)
        if roi is None:
            project_gfc_onto_voxel(gfc, mask, volume)
        else:
            project_gfc_onto_roi(gfc, roi, volume)

            labels = np.unique(roi)
            if 0 in labels:
                labels = labels[labels != 0]
            table = [labels, gfc]

            if out_z:
                table.append(compute_z_score(gfc))
                z = compute_z_score(volume)
                f = f'{out_gfc}{k}_z.nii.gz'
                nib.save(nib.Nifti1Image(z, affine=affine), f)
                logging.info(f'Saved {k} Z score in {f}')

        f = f'{out_gfc}{k}.nii.gz'
        nib.save(nib.Nifti1Image(volume, affine=affine), f)
        logging.info(f'Saved {k} GFC in {f}')

        if out_txt:
            np.savetxt(f'{out_gfc}{k}.txt', np.asarray(table).T)
            logging.info(f'Saved {k} result table in {f}')


if __name__ == '__main__':
    args = get_parsed_args()

    level = logging.WARNING
    if args.debug:
        level = logging.DEBUG
    if args.info:
        level = logging.INFO
    if args.warn:
        level = logging.WARN
    if args.quiet:
        level = logging.CRITICAL

    logging.getLogger().setLevel(level)

    parameters = {k: v for k, v in vars(args).items() if k != 'func'}
    main(**parameters)

    logging.info('Done :)')
