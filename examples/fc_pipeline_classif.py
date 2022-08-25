"""
=================================
 Functional Connectivity with MNE
=================================
This module is design to compute functional connectivity metrics on
MOABB datasets
"""
# Authors: Sylvain Chevallier <sylvain.chevallier@uvsq.fr>,
#          Marie-Constance Corsi <marie.constance.corsi@gmail.com>
#
# License: BSD (3-clause)

import hashlib
import os.path as osp
import os

from mne import get_config, set_config, set_log_level, EpochsArray
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity import envelope_correlation

import numpy as np
from mne.epochs import BaseEpochs

from sklearn.covariance import ledoit_wolf
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
    cohen_kappa_score,
)
from sklearn.ensemble import StackingClassifier

from pyriemann.estimation import Coherences, Covariances

from tqdm import tqdm


def _compute_fc_subtrial(epoch, delta=1, ratio=0.5,
                         method="coh", fmin=8, fmax=35):
    """Compute single trial functional connectivity (FC)

    Most of the FC estimators are already implemented in mne-python (and used
    here from mne.connectivity.spectral_connectivity and
    mne.connectivity.envelope_correlation).
    The epoch is split into subtrials.

    Parameters
    ----------
    epoch: MNE epoch
        Epoch to process
    delta: float
        length of the subtrial in seconds
    ratio: float, in [0, 1]
        ratio overlap of the sliding windows
    method: string
        FC method to be applied, currently implemented methods are: "coh",
        "plv", "imcoh", "pli", "pli2_unbiased", "wpli", "wpli2_debiased",
        "cov", "plm", "aec"
    fmin: real
        filtering frequency, lowpass, in Hz
    fmax: real
        filtering frequency, highpass, in Hz

    Returns
    -------
    connectivity: array, (nb channels x nb channels)

    #TODO: compare matlab/python plm's output
    The only exception is the Phase Linearity Measurement (PLM). In this case,
    it is a Python version of the ft_connectivity_plm MATLAB code [1] of the
    Fieldtrip toolbox [2], which credits [3], with the "translation" into Python
    made by M.-C. Corsi.

    references
    ----------
    .. [1] https://github.com/fieldtrip/fieldtrip/blob/master/connectivity/ft_connectivity_plm.m  # noqa
    .. [2] R. Oostenveld, P. Fries, E. Maris, J.-M. Schoffelen, and
    R. Oostenveld, "FieldTrip: Open Source Software for Advanced Analysis of MEG,
    EEG, and Invasive Electrophysiological  Data" (2010):
    https://doi.org/10.1155/2011/156869
    .. [3] F. Baselice, A. Sorriso, R. Rucco, and P. Sorrentino, "Phase Linearity
    Measurement: A Novel Index for Brain Functional Connectivity" (2019):
    https://doi.org/10.1109/TMI.2018.2873423
    """
    set_log_level("CRITICAL")
    L = epoch.times[-1] - epoch.times[0]
    sliding = ratio * delta
    # fmt: off
    spectral_met = ["coh", "plv", "imcoh", "pli", "pli2_unbiased",
                    "wpli", "wpli2_debiased", ]
    other_met = ["cov", "plm", "aec"]
    # fmt: on
    if method not in spectral_met + other_met:
        raise NotImplementedError("spectral connectivity is not implemented")

    sfreq, nb_chan = epoch.info["sfreq"], epoch.info["nchan"]
    win = delta * sfreq
    nb_subtrials = int(L * (1 / (sliding + delta) + 1 / delta))
    nbsamples_subtrial = delta * sfreq

    # X, total nb trials over the session(s) x nb channels x nb samples
    X = np.squeeze(epoch.get_data())
    subtrials = np.empty((nb_subtrials, nb_chan, int(win)))

    for i in range(0, nb_subtrials):
        idx_start = int(sfreq * i * sliding)
        idx_stop = int(sfreq * i * sliding + nbsamples_subtrial)
        subtrials[i, :, :] = np.expand_dims(X[:, idx_start:idx_stop], axis=0)
    sub_epoch = EpochsArray(np.squeeze(subtrials), info=epoch.info)
    if method in spectral_met:
        r = spectral_connectivity_epochs(
            sub_epoch,
            method=method,
            mode="multitaper",
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            faverage=True,
            tmin=0,
            mt_adaptive=False,
            n_jobs=1,
        )
        # r = SpectralConnectivity(
        #     data=sub_epoch,
        #     freqs=[fmin, fmax],
        #     n_nodes=nb_chan,
        #     method=method,
        #     spec_method="multitaper",
        #     faverage=True,
        #     tmin=0,
        #     mt_adaptive=False,
        #     n_jobs=1,
        # )
        # r = spectral_connectivity(
        #     sub_epoch,
        #     method=method,
        #     mode="multitaper",
        #     sfreq=sfreq,
        #     fmin=fmin,
        #     fmax=fmax,
        #     faverage=True,
        #     tmin=0,
        #     mt_adaptive=False,
        #     n_jobs=1,
        # )
        # c = np.squeeze(r[0])
        c = r.xarray.to_numpy().reshape((nb_chan, nb_chan))
        c = c + c.T - np.diag(np.diag(c)) + np.identity(nb_chan)
    elif method == "aec":
        # filter in frequency band of interest
        sub_epoch.filter(
            fmin,
            fmax,
            n_jobs=1,
            l_trans_bandwidth=1,  # make sure filter params are the same
            h_trans_bandwidth=1,
        )  # in each band and skip "auto" option.
        # apply hilbert transform first
        h_sub_epoch = sub_epoch.apply_hilbert()
        c = envelope_correlation(h_sub_epoch, verbose=True)
        # by default, combine correlation estimates across epochs by peforming
        # an average output : nb_channels x nb_channels -> no need to rearrange
        # the matrix
    elif method == "cov":
        c = ledoit_wolf(X.T)[0]  # oas ou fast_mcd

    return c


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False


def isPD2(B):
    """Returns true when input is positive-definite, via eigenvalues"""
    if np.any(np.linalg.eigvals(B) < 0.0):
        return False
    else:
        return True


def nearestPD(A, reg=1e-6):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): htttps://doi.org/10.1016/0024-3795(88)90223-6
    """
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        # Regularize if already PD
        ei, ev = np.linalg.eigh(A3)
        if np.min(ei) / np.max(ei) < reg:
            A3 = ev @ np.diag(ei + reg) @ ev.T
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])  # noqa
    k = 1
    while not isPD2(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1

    # Regularize
    ei, ev = np.linalg.eigh(A3)
    if np.min(ei) / np.max(ei) < reg:
        A3 = ev @ np.diag(ei + reg) @ ev.T
    return A3


class FunctionalTransformer(TransformerMixin, BaseEstimator):
    """Getting connectivity features from epoch"""

    def __init__(self, delta=1, ratio=0.5, method="coh", fmin=8, fmax=35):
        self.delta = delta
        self.ratio = ratio
        self.method = method
        self.fmin = fmin
        self.fmax = fmax
        if get_config("MOABB_PREPROCESSED") is None:
            set_config(
                "MOABB_PREPROCESSED",
                osp.join(osp.expanduser("~"), "mne_data", "preprocessing"),
            )
        if not osp.isdir(get_config("MOABB_PREPROCESSED")):
            os.makedirs(get_config("MOABB_PREPROCESSED"))
        self.preproc_dir = get_config("MOABB_PREPROCESSED")
        self.cname = "-".join(
            [
                str(e)
                for e in [
                    self.method,
                    self.delta,
                    self.ratio,
                    self.fmin,
                    self.fmax,
                    ".npz",
                ]
            ]
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # StackingClassifier uses cross_val_predict, that apply transform
        # with dispatch_one_batch, streaming each trial one by one :'(
        # If training on a whole set, cache results otherwise compute
        # fc each time
        if isinstance(X, BaseEpochs):
            if self.method in ["instantaneous", "lagged"]:
                Xfc_temp = Coherences(
                    coh=self.method, fmin=self.fmin, fmax=self.fmax,
                    fs=X.info["sfreq"]
                ).fit_transform(X.get_data())
                Xfc = np.empty(Xfc_temp.shape[:-1], dtype=Xfc_temp.dtype)
                for trial, fc in enumerate(Xfc_temp):
                    Xfc[trial, :, :] = fc.mean(axis=-1)
                return Xfc
            elif self.method == "cov":
                return Covariances(estimator="lwf").fit_transform(X.get_data())

            fcache = hashlib.md5(X.get_data()).hexdigest() + self.cname
            if osp.isfile(fcache):
                return np.load(fcache)["Xfc"]
            else:
                Xfc = np.empty((len(X),
                                X[0].info["nchan"],
                                X[0].info["nchan"]))
                for i in range(len(X)):
                    Xfc[i, :, :] = _compute_fc_subtrial(
                        X[i],
                        delta=self.delta,
                        ratio=self.ratio,
                        method=self.method,
                        fmin=self.fmin,
                        fmax=self.fmax,
                    )

            return Xfc


class EnsureSPD(TransformerMixin, BaseEstimator):
    """Getting connectivity features from mat files"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xspd = np.empty_like(X)
        for i, mat in enumerate(X):
            Xspd[i, :, :] = nearestPD(mat)
        return Xspd

    def fit_transform(self, X, y=None):
        transf = self.transform(X)
        return transf


class Epochs2cov(TransformerMixin, BaseEstimator):
    """extract ndarray from epoch"""

    def __init__(self):
        self.C = Covariances(estimator="lwf")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.C.fit_transform(X.get_data())

    def fit_transform(self, X, y=None):
        return self.C.fit_transform(X.get_data())


class GetData(TransformerMixin, BaseEstimator):
    """Get data for ensemble"""

    def __init__(self, paradigm, dataset, subject):
        self.paradigm = paradigm
        self.dataset = dataset
        self.subject = subject

    def fit(self, X, y=None):
        self.ep_, _, self.metadata_ = self.paradigm.get_data(
            self.dataset, [self.subject], return_epochs=True
        )
        return self

    def transform(self, X):
        return self.ep_[X]

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


def get_results(paradigm, dataset, subject, nchan, fmin, fmax, all_ppl):
    subj_res = []
    _, y, metadata = paradigm.get_data(dataset, [subject], return_epochs=True)
    X = np.arange(len(y))
    for session in tqdm(np.unique(metadata.session), desc="session"):
        ix = metadata.session == session
        cv = StratifiedKFold(5, shuffle=True, random_state=42)
        le = LabelEncoder()
        y_cv = le.fit_transform(y[ix])
        X_ = X[ix]
        y_ = y_cv
        for idx, (train, test) in enumerate(cv.split(X_, y_)):
            for ppn, ppl in tqdm(all_ppl.items(),
                                 total=len(all_ppl), desc="pipelines"):
                cvclf = clone(ppl)
                cvclf.fit(X_[train], y_[train])
                yp = cvclf.predict(X_[test])
                acc = balanced_accuracy_score(y_[test], yp)
                auc = roc_auc_score(y_[test], yp)
                kapp = cohen_kappa_score(y_[test], yp)
                res_info = {
                    "subject": subject,
                    "session": "session_0",
                    "channels": nchan,
                    "n_sessions": 1,
                    "FreqBand": "defaultBand",
                    "dataset": dataset.code,
                    "fmin": fmin,
                    "fmax": fmax,
                    "samples": len(y_),
                    "time": 0.0,
                    "split": idx,
                }
                res = {
                    "score": auc,
                    "kappa": kapp,
                    "accuracy": acc,
                    "pipeline": ppn,
                    "n_dr": nchan,
                    "thres": 0,
                    **res_info,
                }
                subj_res.append(res)
                if isinstance(ppl, StackingClassifier):
                    for est_n, est_p in cvclf.named_estimators_.items():
                        thres, n_dr = 0, nchan
                        ype = est_p.predict(X_[test])
                        acc = balanced_accuracy_score(y_[test], ype)
                        auc = roc_auc_score(y_[test], ype)
                        kapp = cohen_kappa_score(y_[test], ype)
                        res = {
                            "score": auc,
                            "kappa": kapp,
                            "accuracy": acc,
                            "pipeline": est_n,
                            "thres": thres,
                            "n_dr": n_dr,
                            **res_info,
                        }
                        subj_res.append(res)
    return subj_res
