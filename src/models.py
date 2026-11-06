"""Section 3.6: tuned logistic regression (main) and XGBoost (exploratory).
No resampling and no class weighting anywhere (Sec 2.6)."""
import warnings

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.preprocess import GroupPreprocessor

C_GRID = np.logspace(-3, 2, 10)
N_TUNE_FOLDS = 3
N_XGB_CONFIGS = 30
XGB_SPACE = {
    "xgb__n_estimators": randint(100, 501),
    "xgb__max_depth": randint(2, 7),
    "xgb__learning_rate": loguniform(0.01, 0.3),
    "xgb__subsample": uniform(0.6, 0.4),
    "xgb__colsample_bytree": uniform(0.5, 0.5),
    "xgb__min_child_weight": randint(1, 11),
    "xgb__reg_lambda": loguniform(0.1, 10),
}


def _cv(seed):
    return StratifiedKFold(N_TUNE_FOLDS, shuffle=True, random_state=seed)


def tune_lr(X, y, target, groups, seed):
    """Tune C over 10 log-spaced values in [1e-3, 1e2] by inner 3-fold AUPRC on D_fit,
    then refit on all of D_fit. Ties go to the smallest C."""
    pipe = Pipeline([("pp", GroupPreprocessor(target, tuple(groups))),
                     ("lr", LogisticRegression(max_iter=5000))])
    gs = GridSearchCV(pipe, {"lr__C": C_GRID}, scoring="average_precision",
                      cv=_cv(seed), refit=True, n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gs.fit(X, np.asarray(y, int))
    return gs.best_estimator_, float(gs.best_params_["lr__C"])


def tune_xgb(X, y, target, groups, seed):
    from xgboost import XGBClassifier
    pipe = Pipeline([("pp", GroupPreprocessor(target, tuple(groups), scale=False)),
                     ("xgb", XGBClassifier(tree_method="hist", n_jobs=1, random_state=seed,
                                           eval_metric="logloss", verbosity=0))])
    rs = RandomizedSearchCV(pipe, XGB_SPACE, n_iter=N_XGB_CONFIGS, scoring="average_precision",
                            cv=_cv(seed), refit=True, random_state=seed, n_jobs=1)
    rs.fit(X, np.asarray(y, int))
    return rs.best_estimator_, {k.split("__")[1]: v for k, v in rs.best_params_.items()}
