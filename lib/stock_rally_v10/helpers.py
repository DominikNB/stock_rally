"""stock_rally_v10 — Hilfsfunktionen (Pipeline-Modul)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _strip_tz(series):
    """Normalize DatetimeIndex or datetime Series to tz-naive midnight."""
    if hasattr(series, 'dt'):
        if series.dt.tz is not None:
            series = series.dt.tz_convert(None)
        return series.dt.normalize()
    if isinstance(series, pd.DatetimeIndex):
        if series.tz is not None:
            series = series.tz_convert(None)
        return series.normalize()
    return series


def make_focal_objective(gamma, alpha):
    """
    Returns a custom objective function for XGBoost/LightGBM implementing
    Focal Loss with given focusing parameter gamma and class weight alpha.

    L = -alpha_t * (1 - p_t)^gamma * log(p_t)
    where p_t = p if y=1 else (1-p), alpha_t = alpha if y=1 else (1-alpha)
    """
    def focal_obj(y_pred, dtrain):
        y_true = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))          # sigmoid
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # Gradient
        grad = (
            y_true * alpha * (1 - p) ** gamma
            * (gamma * p * np.log(p) + p - 1)
            + (1 - y_true) * (1 - alpha) * p ** gamma
            * (p - gamma * (1 - p) * np.log(1 - p))
        )

        # Hessian (weighted logistic approximation)
        p_t   = np.where(y_true == 1, p, 1 - p)
        a_t   = np.where(y_true == 1, alpha, 1 - alpha)
        hess  = np.maximum(a_t * (1 - p_t) ** gamma * p * (1 - p), 1e-7)
        return grad, hess

    return focal_obj


def make_focal_objective_lgb(gamma, alpha):
    """LightGBM version of focal loss objective (raw logit input)."""
    def focal_obj_lgb(y_pred, dataset):
        y_true = dataset.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)

        grad = (
            y_true * alpha * (1 - p) ** gamma
            * (gamma * p * np.log(p) + p - 1)
            + (1 - y_true) * (1 - alpha) * p ** gamma
            * (p - gamma * (1 - p) * np.log(1 - p))
        )

        p_t  = np.where(y_true == 1, p, 1 - p)
        a_t  = np.where(y_true == 1, alpha, 1 - alpha)
        hess = np.maximum(a_t * (1 - p_t) ** gamma * p * (1 - p), 1e-7)
        return grad, hess

    return focal_obj_lgb


print('Helpers defined.')