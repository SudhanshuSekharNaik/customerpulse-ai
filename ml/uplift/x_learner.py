"""X-Learner Uplift Modeling algorithm for CustomerPulse AI.
Implements the 4-step meta-learner algorithm for heterogeneous treatment effect estimation:
1. Estimate response functions mu0(x) and mu1(x).
2. Impute counterfactuals D1 = Y1 - mu0(X1) and D0 = mu1(X0) - Y0.
3. Train two second-stage models tau1(x) and tau0(x) predicting imputed treatment effects.
4. Combine via propensity score: tau(x) = e(x)*tau0(x) + (1 - e(x))*tau1(x).
"""

from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression


class XLearner:
    """Purpose-built X-Learner for heterogeneous treatment effect estimation."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05, max_depth: int = 4):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        # Stage 1 models
        self.model_mu0 = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, verbose=-1)
        self.model_mu1 = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, verbose=-1)
        
        # Propensity model
        self.propensity_model = LogisticRegression(random_state=42, max_iter=200)
        
        # Stage 2 models (regressors on imputed treatment effects)
        self.model_tau0 = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, verbose=-1)
        self.model_tau1 = lgb.LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, verbose=-1)

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        """Fit the X-Learner on feature matrix X, binary treatment, and binary conversion outcome y."""
        X_0 = X[treatment == 0]
        y_0 = y[treatment == 0]
        X_1 = X[treatment == 1]
        y_1 = y[treatment == 1]

        # 1. Fit Stage 1 models
        self.model_mu0.fit(X_0, y_0)
        self.model_mu1.fit(X_1, y_1)

        # 2. Fit Propensity model e(x) = P(T=1|X)
        self.propensity_model.fit(X, treatment)

        # 3. Impute counterfactual treatment effects
        # D_1 = Y_1 - mu0(X_1)
        mu0_on_1 = self.model_mu0.predict_proba(X_1)[:, 1]
        D_1 = y_1 - mu0_on_1

        # D_0 = mu1(X_0) - Y_0
        mu1_on_0 = self.model_mu1.predict_proba(X_0)[:, 1]
        D_0 = mu1_on_0 - y_0

        # 4. Fit Stage 2 models on imputed effects
        self.model_tau1.fit(X_1, D_1)
        self.model_tau0.fit(X_0, D_0)

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Predict individual treatment effect (CATE / uplift) for feature matrix X."""
        propensities = self.propensity_model.predict_proba(X)[:, 1]
        tau0_preds = self.model_tau0.predict(X)
        tau1_preds = self.model_tau1.predict(X)

        # Weighted combination: tau(x) = e(x)*tau0(x) + (1 - e(x))*tau1(x)
        uplift = propensities * tau0_preds + (1.0 - propensities) * tau1_preds
        return uplift


class TwoModelUplift:
    """Two-Model Baseline: P(Y=1 | T=1, X) - P(Y=1 | T=0, X)."""

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.05, max_depth: int = 4):
        self.model_control = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, verbose=-1)
        self.model_treated = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=42, verbose=-1)

    def fit(self, X: np.ndarray, treatment: np.ndarray, y: np.ndarray):
        X_0, y_0 = X[treatment == 0], y[treatment == 0]
        X_1, y_1 = X[treatment == 1], y[treatment == 1]
        self.model_control.fit(X_0, y_0)
        self.model_treated.fit(X_1, y_1)

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        p_control = self.model_control.predict_proba(X)[:, 1]
        p_treated = self.model_treated.predict_proba(X)[:, 1]
        return p_treated - p_control
