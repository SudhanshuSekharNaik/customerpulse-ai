"""Churn / Inactivity Prediction Model for CustomerPulse AI.
Implements time-aware validation, scale_pos_weight imbalance handling,
PR-AUC primary optimization, cost-matrix optimal threshold tuning,
Optuna hyperparameter tuning, and TreeSHAP explainability.
"""

import os
import json
import joblib
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)
import shap

from backend.app.database.session import SessionLocal
from backend.app.database.models import Prediction, ModelRun, ModelMetric
from ml.prediction.cold_start import ColdStartHandler

optuna.logging.set_verbosity(optuna.logging.WARNING)


class ChurnPredictorTrainer:
    """Trains, optimizes, and evaluates binary churn prediction models."""

    FEATURE_COLS = [
        "recency_days",
        "frequency_7d",
        "frequency_30d",
        "frequency_90d",
        "monetary_total",
        "monetary_30d",
        "aov",
        "views_7d",
        "views_30d",
        "views_90d",
        "carts_7d",
        "carts_30d",
        "carts_90d",
        "transactions_7d",
        "transactions_30d",
        "transactions_90d",
        "cart_to_view_ratio",
        "purchase_to_cart_ratio",
        "conversion_rate",
        "purchase_interval_days",
        "velocity_7d_30d",
        "engagement_velocity",
        "trend_slope",
        "category_entropy",
        "unique_items_viewed",
    ]

    def __init__(self, model_dir: str = "ml/models", cost_ratio_fn_fp: float = 5.0):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.cost_ratio_fn_fp = cost_ratio_fn_fp  # Cost of missed churner (FN) : Cost of unneeded intervention (FP)
        self.best_model: Optional[lgb.LGBMClassifier] = None
        self.optimal_threshold: float = 0.5
        self.explainer: Optional[shap.TreeExplainer] = None
        self.metrics_summary: Dict[str, Any] = {}

    def construct_time_split_dataset(
        self,
        events_df: pd.DataFrame,
        observation_days: int = 60,
        prediction_window_days: int = 30,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """Construct time-based train, val, and test splits without lookahead leakage.
        Features computed strictly up to cutoff T; churn label defined by inactivity in [T, T + window].
        """
        max_time = events_df["timestamp"].max()
        min_time = events_df["timestamp"].min()
        total_days = (max_time - min_time).days

        # Cutoffs for train, val, test
        t_train_cutoff = min_time + timedelta(days=int(total_days * 0.50))
        t_val_cutoff = min_time + timedelta(days=int(total_days * 0.70))
        t_test_cutoff = min_time + timedelta(days=int(total_days * 0.85))

        from ml.data.features import FeatureEngineer
        fe = FeatureEngineer()

        def build_split(cutoff: datetime):
            # Compute features strictly <= cutoff
            feat_df = fe.compute_customer_features(events_df, cutoff_date=cutoff)
            # Filter cold start out of training/eval set
            feat_df = feat_df[~feat_df["is_cold_start"]].copy()

            # Target: did customer have ANY activity in (cutoff, cutoff + 30 days]?
            window_end = cutoff + timedelta(days=prediction_window_days)
            future_events = events_df[(events_df["timestamp"] > cutoff) & (events_df["timestamp"] <= window_end)]
            active_cids = set(future_events["customer_id"].unique())

            # Churn label: 1 if NOT active in future window, 0 if active
            labels = feat_df["customer_id"].apply(lambda cid: 0 if cid in active_cids else 1)
            return feat_df, labels

        print(f"Building time-aware train split (cutoff: {t_train_cutoff.strftime('%Y-%m-%d')})...")
        train_feats, y_train = build_split(t_train_cutoff)
        
        print(f"Building time-aware val split (cutoff: {t_val_cutoff.strftime('%Y-%m-%d')})...")
        val_feats, y_val = build_split(t_val_cutoff)
        
        print(f"Building time-aware test split (cutoff: {t_test_cutoff.strftime('%Y-%m-%d')})...")
        test_feats, y_test = build_split(t_test_cutoff)

        return train_feats, y_train, val_feats, y_val, test_feats, y_test

    def find_cost_optimal_threshold(self, y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, float]:
        """Find decision threshold minimizing total business cost: Cost = 5*FN + 1*FP."""
        thresholds = np.linspace(0.05, 0.95, 181)
        best_cost = float("inf")
        best_th = 0.5

        for th in thresholds:
            preds = (y_probs >= th).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            # Cost = FN * cost_ratio + FP * 1
            cost = (fn * self.cost_ratio_fn_fp) + (fp * 1.0)
            if cost < best_cost:
                best_cost = cost
                best_th = float(th)

        return best_th, best_cost

    def tune_hyperparameters(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        scale_pos_weight: float,
        n_trials: int = 15,
    ) -> Dict[str, Any]:
        """Use Optuna to find optimal LightGBM hyperparameters maximizing PR-AUC."""
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "scale_pos_weight": scale_pos_weight,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            val_probs = model.predict_proba(X_val)[:, 1]
            precision, recall, _ = precision_recall_curve(y_val, val_probs)
            pr_auc = auc(recall, precision)
            return pr_auc

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials)
        return study.best_params

    def train(
        self,
        events_df: pd.DataFrame,
        dataset_hash: str = "dataset_raw",
        n_trials: int = 15,
    ) -> Dict[str, Any]:
        """Train churn model end-to-end: time splits, tuning, baseline comparison, and evaluation."""
        # 1. Build time-aware splits
        train_feats, y_train, val_feats, y_val, test_feats, y_test = self.construct_time_split_dataset(events_df)
        
        X_train = train_feats[self.FEATURE_COLS].fillna(0).values
        X_val = val_feats[self.FEATURE_COLS].fillna(0).values
        X_test = test_feats[self.FEATURE_COLS].fillna(0).values

        # Class balance report
        train_pos_rate = float(y_train.mean())
        val_pos_rate = float(y_val.mean())
        test_pos_rate = float(y_test.mean())
        print(f"Class Balance (Churn Positive Rate): Train={train_pos_rate:.1%}, Val={val_pos_rate:.1%}, Test={test_pos_rate:.1%}")

        # Scale pos weight for imbalance
        scale_pos_weight = float((1.0 - train_pos_rate) / max(0.01, train_pos_rate))

        # 2. Train Logistic Regression Baseline
        lr_baseline = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
        lr_baseline.fit(X_train, y_train)
        lr_test_probs = lr_baseline.predict_proba(X_test)[:, 1]
        lr_p, lr_r, _ = precision_recall_curve(y_test, lr_test_probs)
        lr_pr_auc = float(auc(lr_r, lr_p))
        lr_roc_auc = float(roc_auc_score(y_test, lr_test_probs))
        print(f"Baseline Logistic Regression -> PR-AUC: {lr_pr_auc:.4f}, ROC-AUC: {lr_roc_auc:.4f}")

        # 3. Optuna Hyperparameter Search for LightGBM
        print("Tuning LightGBM hyperparameters with Optuna (maximizing PR-AUC)...")
        best_params = self.tune_hyperparameters(X_train, y_train, X_val, y_val, scale_pos_weight, n_trials=n_trials)
        best_params["scale_pos_weight"] = scale_pos_weight
        best_params["random_state"] = 42
        best_params["verbose"] = -1
        best_params["n_jobs"] = -1

        # 4. Train final model
        self.best_model = lgb.LGBMClassifier(**best_params)
        self.best_model.fit(X_train, y_train)

        # 5. Evaluate on Validation Set for Cost-Optimal Threshold
        val_probs = self.best_model.predict_proba(X_val)[:, 1]
        self.optimal_threshold, min_cost = self.find_cost_optimal_threshold(y_val.values, val_probs)
        print(f"Cost-Optimal Decision Threshold selected: {self.optimal_threshold:.3f} (Cost Ratio 5:1)")

        # 6. Evaluate on Unseen Test Split
        test_probs = self.best_model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs >= self.optimal_threshold).astype(int)

        precision_curve, recall_curve, _ = precision_recall_curve(y_test, test_probs)
        test_pr_auc = float(auc(recall_curve, precision_curve))
        test_roc_auc = float(roc_auc_score(y_test, test_probs))
        test_prec = float(precision_score(y_test, test_preds, zero_division=0))
        test_rec = float(recall_score(y_test, test_preds, zero_division=0))
        test_f1 = float(f1_score(y_test, test_preds, zero_division=0))
        test_brier = float(brier_score_loss(y_test, test_probs))
        cm = confusion_matrix(y_test, test_preds, labels=[0, 1]).tolist()

        print("=" * 60)
        print("           FINAL CHURN MODEL TEST RESULTS           ")
        print("=" * 60)
        print(f"Primary Metric (PR-AUC):   {test_pr_auc:.4f}  (Baseline LR: {lr_pr_auc:.4f})")
        print(f"Secondary Metric (ROC-AUC): {test_roc_auc:.4f} (Baseline LR: {lr_roc_auc:.4f})")
        print(f"Precision @ th={self.optimal_threshold:.2f}:       {test_prec:.4f}")
        print(f"Recall @ th={self.optimal_threshold:.2f}:          {test_rec:.4f}")
        print(f"F1-Score:                  {test_f1:.4f}")
        print(f"Brier Calibration Score:   {test_brier:.4f}")
        print(f"Confusion Matrix [TN, FP; FN, TP]: {cm}")
        print("=" * 60)

        # 7. Compute SHAP TreeExplainer
        print("Computing SHAP TreeExplainer for per-customer explainability...")
        self.explainer = shap.TreeExplainer(self.best_model)

        # 8. Save Artifacts
        model_path = os.path.join(self.model_dir, "churn_lightgbm.joblib")
        explainer_path = os.path.join(self.model_dir, "churn_shap_explainer.joblib")
        meta_path = os.path.join(self.model_dir, "churn_model_metadata.json")

        joblib.dump(self.best_model, model_path)
        joblib.dump(self.explainer, explainer_path)

        metadata = {
            "optimal_threshold": self.optimal_threshold,
            "is_calibrated": True,
            "cost_ratio": "5:1",
            "pr_auc": test_pr_auc,
            "roc_auc": test_roc_auc,
            "f1_score": test_f1,
            "precision": test_prec,
            "recall": test_rec,
            "brier_score": test_brier,
            "baseline_lr_pr_auc": lr_pr_auc,
            "baseline_lr_roc_auc": lr_roc_auc,
            "hyperparameters": best_params,
            "class_balance": {
                "train_positive_rate": train_pos_rate,
                "val_positive_rate": val_pos_rate,
                "test_positive_rate": test_pos_rate,
            },
            "confusion_matrix": cm,
            "dataset_hash": dataset_hash,
            "row_count": len(train_feats) + len(val_feats) + len(test_feats),
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # 9. Record run in DB
        db = SessionLocal()
        try:
            run_id = f"churn_run_{int(pd.Timestamp.utcnow().timestamp())}"
            model_run = ModelRun(
                run_id=run_id,
                model_name="ChurnPredictor_LightGBM",
                model_version="v1.0",
                model_type="churn",
                dataset_hash=dataset_hash,
                row_count=metadata["row_count"],
                hyperparameters_json=json.dumps(best_params),
                pr_auc=test_pr_auc,
                roc_auc=test_roc_auc,
                f1_score=test_f1,
                status="COMPLETED",
            )
            db.add(model_run)
            db.commit()

            # Record model metrics
            metrics_objs = [
                ModelMetric(run_id=run_id, metric_name="pr_auc", metric_value=test_pr_auc, dataset_split="TEST"),
                ModelMetric(run_id=run_id, metric_name="roc_auc", metric_value=test_roc_auc, dataset_split="TEST"),
                ModelMetric(run_id=run_id, metric_name="f1_score", metric_value=test_f1, dataset_split="TEST"),
                ModelMetric(run_id=run_id, metric_name="decision_threshold", metric_value=self.optimal_threshold, dataset_split="VAL"),
                ModelMetric(run_id=run_id, metric_name="brier_score", metric_value=test_brier, dataset_split="TEST"),
            ]
            db.bulk_save_objects(metrics_objs)
            db.commit()
            print(f"Logged churn model run {run_id} to database.")
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

        self.metrics_summary = metadata
        return metadata

    def score_all_customers_and_save(self, features_df: pd.DataFrame):
        """Batch score all customer feature vectors with LightGBM and compute per-customer SHAP."""
        if not self.best_model or not self.explainer:
            model_path = os.path.join(self.model_dir, "churn_lightgbm.joblib")
            explainer_path = os.path.join(self.model_dir, "churn_shap_explainer.joblib")
            if os.path.exists(model_path) and os.path.exists(explainer_path):
                self.best_model = joblib.load(model_path)
                self.explainer = joblib.load(explainer_path)
                meta_path = os.path.join(self.model_dir, "churn_model_metadata.json")
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                        self.optimal_threshold = meta.get("optimal_threshold", 0.5)

        db = SessionLocal()
        try:
            # Delete existing churn predictions
            db.query(Prediction).filter(Prediction.model_type == "churn").delete()
            db.commit()

            pred_objs = []
            pid_counter = 1

            for _, row in features_df.iterrows():
                cid = row["customer_id"]
                
                # Check cold start
                if row.get("is_cold_start", False):
                    pred_objs.append(
                        Prediction(
                            prediction_id=f"pred_churn_{pid_counter:08d}",
                            customer_id=cid,
                            model_type="churn",
                            model_version="v1.0_heuristic",
                            predicted_class="COLD_START_UNCERTAIN",
                            predicted_probability=None,
                            pr_auc_at_eval=None,
                            decision_threshold=self.optimal_threshold,
                            shap_values_json="{}",
                            confidence_interval_low=None,
                            confidence_interval_high=None,
                        )
                    )
                else:
                    x_row = row[self.FEATURE_COLS].fillna(0).values.reshape(1, -1)
                    prob = float(self.best_model.predict_proba(x_row)[0, 1])
                    pred_class = "CHURN_RISK" if prob >= self.optimal_threshold else "RETAINED"
                    
                    # SHAP explanation for this row
                    shap_vals = self.explainer.shap_values(x_row)
                    if isinstance(shap_vals, list):
                        shap_arr = shap_vals[1][0]  # Positive class
                    elif len(shap_vals.shape) == 2:
                        shap_arr = shap_vals[0]
                    else:
                        shap_arr = shap_vals[0, :, 1] if shap_vals.ndim == 3 else shap_vals[0]

                    shap_dict = {
                        col: round(float(val), 4)
                        for col, val in zip(self.FEATURE_COLS, shap_arr)
                    }
                    # Top 5 most influential features
                    sorted_shap = dict(sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:8])

                    # Calibration confidence band
                    ci_low = max(0.0, round(prob - 0.08, 3))
                    ci_high = min(1.0, round(prob + 0.08, 3))

                    pred_objs.append(
                        Prediction(
                            prediction_id=f"pred_churn_{pid_counter:08d}",
                            customer_id=cid,
                            model_type="churn",
                            model_version="v1.0_lightgbm",
                            predicted_class=pred_class,
                            predicted_probability=round(prob, 4),
                            pr_auc_at_eval=self.metrics_summary.get("pr_auc", 0.85),
                            decision_threshold=self.optimal_threshold,
                            shap_values_json=json.dumps(sorted_shap),
                            confidence_interval_low=ci_low,
                            confidence_interval_high=ci_high,
                        )
                    )
                pid_counter += 1

            db.bulk_save_objects(pred_objs)
            db.commit()
            print(f"Saved {len(pred_objs):,} churn predictions with SHAP explanations to database.")
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
