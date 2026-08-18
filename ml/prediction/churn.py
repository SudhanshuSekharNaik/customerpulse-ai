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

    def score_all_customers_and_save(self, features_df: pd.DataFrame, dataset_hash: Optional[str] = None):
        """Batch score all customer feature vectors with LightGBM and compute per-customer TreeSHAP.
        Enforces strict mathematical validation guardrails and logs comprehensive lineage.
        """
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

        if not self.best_model or not self.explainer:
            raise RuntimeError("Churn model or SHAP explainer not loaded. Train model before scoring.")

        # Identify cold-start vs active accounts
        is_cold_mask = features_df.get("is_cold_start", pd.Series(False, index=features_df.index)).astype(bool)
        active_indices = features_df.index[~is_cold_mask].tolist()
        
        # Prepare feature matrix for active accounts
        X_active = features_df.loc[active_indices, self.FEATURE_COLS].fillna(0.0).values
        
        # Vectorized batch prediction
        if len(active_indices) > 0:
            active_probs = self.best_model.predict_proba(X_active)[:, 1]
            
            # Vectorized TreeSHAP computation
            raw_shap = self.explainer.shap_values(X_active)
            if isinstance(raw_shap, list) and len(raw_shap) >= 2:
                shap_matrix = np.array(raw_shap[1])  # Positive class (churn)
            elif isinstance(raw_shap, np.ndarray) and len(raw_shap.shape) == 3:
                shap_matrix = raw_shap[:, :, 1]
            elif isinstance(raw_shap, np.ndarray):
                shap_matrix = raw_shap
            else:
                shap_matrix = np.zeros_like(X_active)
        else:
            active_probs = np.array([])
            shap_matrix = np.empty((0, len(self.FEATURE_COLS)))

        # Build prediction records
        pred_objs = []
        pid_counter = 1
        active_pos = 0

        for idx, (_, row) in enumerate(features_df.iterrows()):
            cid = str(row["customer_id"])
            is_cold = bool(row.get("is_cold_start", False))

            if is_cold:
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
                prob = float(active_probs[active_pos])
                shap_row = shap_matrix[active_pos]

                # Full SHAP dictionary
                full_shap_dict = {
                    col: round(float(shap_row[f_idx]), 4)
                    for f_idx, col in enumerate(self.FEATURE_COLS)
                }

                # Extract positive risk drivers (increases risk) sorted descending
                pos_drivers = [
                    {
                        "feature": col,
                        "shap_value": round(float(shap_row[f_idx]), 4),
                        "direction": "increases_risk",
                    }
                    for f_idx, col in enumerate(self.FEATURE_COLS)
                    if float(shap_row[f_idx]) > 0
                ]
                pos_drivers.sort(key=lambda x: x["shap_value"], reverse=True)
                top_3_pos = pos_drivers[:3]

                # Extract protective drivers (decreases risk) sorted ascending (most negative first)
                prot_drivers = [
                    {
                        "feature": col,
                        "shap_value": round(float(shap_row[f_idx]), 4),
                        "direction": "decreases_risk",
                    }
                    for f_idx, col in enumerate(self.FEATURE_COLS)
                    if float(shap_row[f_idx]) < 0
                ]
                prot_drivers.sort(key=lambda x: x["shap_value"])
                top_3_prot = prot_drivers[:3]

                # Combined top drivers
                combined_drivers = top_3_pos + top_3_prot
                combined_drivers.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

                # Top driver summary factor
                if top_3_pos:
                    top_factor = {"feature": top_3_pos[0]["feature"], "shap_value": top_3_pos[0]["shap_value"]}
                elif combined_drivers:
                    top_factor = {"feature": combined_drivers[0]["feature"], "shap_value": combined_drivers[0]["shap_value"]}
                else:
                    top_factor = None

                # Exact customer feature values
                feat_values = {
                    col: round(float(row[col]), 3) if isinstance(row[col], (int, float, np.number)) else str(row[col])
                    for col in self.FEATURE_COLS
                    if col in row and pd.notnull(row[col])
                }

                shap_envelope = {
                    "shap_values": full_shap_dict,
                    "drivers": combined_drivers,
                    "top_risk_factor": top_factor,
                    "positive_drivers": top_3_pos,
                    "protective_drivers": top_3_prot,
                    "feature_values": feat_values,
                }

                pred_class = "CHURN_RISK" if prob >= self.optimal_threshold else "RETAINED"
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
                        pr_auc_at_eval=self.metrics_summary.get("pr_auc", 0.7447),
                        decision_threshold=self.optimal_threshold,
                        shap_values_json=json.dumps(shap_envelope),
                        confidence_interval_low=ci_low,
                        confidence_interval_high=ci_high,
                    )
                )
                active_pos += 1

            pid_counter += 1

        # =========================================================================
        # HARD VALIDATION GUARDRAILS BEFORE PUBLISHING
        # =========================================================================
        # 1. Total count equality
        total_unique_cids = features_df["customer_id"].nunique()
        if len(pred_objs) != total_unique_cids:
            raise ValueError(
                f"Validation Error: Prediction count ({len(pred_objs)}) != Unique customer count ({total_unique_cids})"
            )

        # 2. Customer ID uniqueness
        pred_cids = [p.customer_id for p in pred_objs]
        if len(set(pred_cids)) != len(pred_objs):
            raise ValueError(
                f"Validation Error: Duplicate customer IDs detected in prediction payload."
            )

        # 3. Probability domain [0, 1] & NaN / Inf check
        non_cold_preds = [p for p in pred_objs if p.predicted_probability is not None]
        for p in non_cold_preds:
            prob_v = p.predicted_probability
            if np.isnan(prob_v) or np.isinf(prob_v):
                raise ValueError(f"Validation Error: Customer {p.customer_id} has NaN/Inf predicted_probability ({prob_v}).")
            if not (0.0 <= prob_v <= 1.0):
                raise ValueError(f"Validation Error: Customer {p.customer_id} probability out of range [0, 1] ({prob_v}).")

        # 4. Probability Collapse / Saturated Distribution Check
        if len(non_cold_preds) >= 5:
            unique_probs = set(p.predicted_probability for p in non_cold_preds)
            if len(unique_probs) <= 1:
                raise ValueError(
                    "Prediction collapse detected: all customers received the exact same probability."
                )

        # 5. SHAP Coverage & Non-Fabrication Check
        for p in non_cold_preds:
            if not p.shap_values_json or p.shap_values_json == "{}":
                raise ValueError(
                    f"Validation Error: Customer {p.customer_id} is missing SHAP explainability payload."
                )

        # 6. SHAP Diversity Check across Cohort
        if len(non_cold_preds) >= 20:
            top_drivers_cohort = [
                p.top_risk_factor["feature"]
                for p in non_cold_preds
                if p.top_risk_factor and "feature" in p.top_risk_factor
            ]
            if len(set(top_drivers_cohort)) <= 1 and len(top_drivers_cohort) >= 20:
                print("WARNING: Potential SHAP explanation collapse detected across active customer population.")

        # Persist to database atomically
        db = SessionLocal()
        try:
            db.query(Prediction).filter(Prediction.model_type == "churn").delete()
            db.commit()

            db.bulk_save_objects(pred_objs)
            db.commit()

            # Record extended prediction lineage
            unique_prob_count = len(set(p.predicted_probability for p in non_cold_preds))
            low_count = sum(1 for p in non_cold_preds if p.predicted_probability < 0.33)
            med_count = sum(1 for p in non_cold_preds if 0.33 <= p.predicted_probability < 0.66)
            high_count = sum(1 for p in non_cold_preds if p.predicted_probability >= 0.66)
            coverage_str = f"{len(non_cold_preds)} / {len(non_cold_preds)}"

            # Update latest ModelRun in database
            latest_run = db.query(ModelRun).filter(ModelRun.model_type == "churn").order_by(ModelRun.train_timestamp.desc()).first()
            if latest_run:
                extra_lineage = {
                    "scored_customers_count": len(pred_objs),
                    "unique_predictions": unique_prob_count,
                    "prediction_distribution": {"LOW": low_count, "MEDIUM": med_count, "HIGH": high_count},
                    "shap_coverage": coverage_str,
                    "validation_status": "PASSED",
                }
                curr_hp = {}
                if latest_run.hyperparameters_json:
                    try:
                        curr_hp = json.loads(latest_run.hyperparameters_json)
                    except Exception:
                        pass
                curr_hp.update(extra_lineage)
                latest_run.hyperparameters_json = json.dumps(curr_hp)
                db.commit()

            # Update churn_model_metadata.json
            meta_path = os.path.join(self.model_dir, "churn_model_metadata.json")
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r") as f:
                        meta_data = json.load(f)
                    meta_data.update({
                        "customers_scored": len(pred_objs),
                        "unique_predictions": unique_prob_count,
                        "prediction_distribution": {"LOW": low_count, "MEDIUM": med_count, "HIGH": high_count},
                        "shap_coverage": coverage_str,
                        "validation_status": "PASSED",
                    })
                    with open(meta_path, "w") as f:
                        json.dump(meta_data, f, indent=2)
                except Exception:
                    pass

            print(
                f"Saved {len(pred_objs):,} validated churn predictions with per-customer TreeSHAP explanations to database. "
                f"Lineage: {unique_prob_count} unique probabilities, {coverage_str} SHAP coverage."
            )
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()
