"""
Optional: übliche Importe (NumPy, pandas, sklearn, …) als Attribute auf ``config`` legen.

Wird von ``bind_step_functions()`` importiert; legt z. B. ``cfg.np`` / ``cfg.pd`` an,
falls späterer Pipeline-Code diese Namen über den gemeinsamen ``config``-Namespace erwartet.
"""
from __future__ import annotations

import base64
import io
import json
import pickle
import subprocess
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import requests
import shap
import ta
import xgboost as xgb
import yfinance as yf
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

from lib.stock_rally_v10 import config as cfg

_NAMES = [
    "ThreadPoolExecutor",
    "datetime",
    "timedelta",
    "timezone",
    "Path",
    "base64",
    "io",
    "json",
    "pickle",
    "subprocess",
    "time",
    "warnings",
    "np",
    "pd",
    "requests",
    "yf",
    "ta",
    "lgb",
    "optuna",
    "shap",
    "xgb",
    "ExtraTreesClassifier",
    "RandomForestClassifier",
    "LogisticRegression",
    "average_precision_score",
    "f1_score",
    "precision_recall_curve",
    "precision_score",
    "recall_score",
    "Pipeline",
    "StandardScaler",
    "mdates",
    "plt",
]


def apply_cell1_imports_to_config() -> None:
    env = {k: globals()[k] for k in _NAMES if k in globals()}
    for name, obj in env.items():
        setattr(cfg, name, obj)


apply_cell1_imports_to_config()
