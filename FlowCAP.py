"""
Queen Flow ML - Signal Filter with Full GUI + Detailed Metrics
===============================================================
Complete signal filter with detailed training performance reporting.
Includes: per-model metrics, confusion matrix, ROC curve, feature importance,
probability calibration, and comprehensive training statistics.
"""

import os
import csv
import json
import time
import queue
import threading
import warnings
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import chardet

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, roc_curve, average_precision_score,
    log_loss, brier_score_loss, precision_recall_curve, classification_report
)

import xgboost as xgb
from catboost import CatBoostClassifier

import joblib
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_FEATURE_COLUMNS = [
    "atr_value", "atr_percent", "adx_value", "di_plus", "di_minus",
    "ema_value", "ema_distance", "htf_ema_value", "htf_ema_distance",
    "rsi_value", "macd_value", "macd_signal",
    "bb_upper", "bb_lower", "bb_middle", "bb_position", "volume_ratio",
    "candle_body", "candle_range", "upper_wick", "lower_wick",
    "trend_strength", "volatility", "price_position",
    "key_value", "sl_atr_multiplier",
]

TARGET_COLUMN = "target_hit_tp1"

TRAINING_FEATURES = [
    "atr_value", "atr_percent", "adx_value", "di_plus", "di_minus",
    "ema_value", "ema_distance", "htf_ema_value", "htf_ema_distance",
    "rsi_value", "macd_value", "macd_signal",
    "bb_upper", "bb_lower", "bb_middle", "bb_position", "volume_ratio",
    "candle_body", "candle_range", "upper_wick", "lower_wick",
    "trend_strength", "volatility", "price_position",
]

LIVE_SIGNAL_PREFIX_COLS = [
    "signal_id", "timestamp", "symbol", "timeframe", "direction",
    "entry_price", "sl_price", "tp1_price", "tp2_price", "tp3_price",
]

DEFAULT_THRESHOLD = 0.60
DEFAULT_PROBABILITY_MULTIPLIER = 1.19  # Default multiplier

# Minimum ensemble test AUC we consider "there might be real signal here".
MIN_MEANINGFUL_AUC = 0.55
MIN_MEANINGFUL_MCC = 0.05

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(APP_DIR, "models")
LOGS_DIR = os.path.join(APP_DIR, "logs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

LIVE_SIGNAL_FILE = "FlowML_signal.csv"
EA_PREDICTION_FILE = "FlowML_output_prediction.csv"

# Header row written to the EA prediction output CSV.
EA_PREDICTION_HEADER = [
    "signal_id", "timestamp", "symbol", "direction",
    "entry_price", "sl_price", "tp1_price", "tp2_price", "tp3_price",
    "probability", "recommendation",
]

# --------------------------------------------------------------------------- #
# Theme colors
# --------------------------------------------------------------------------- #
COLORS = {
    "bg_dark": "#0d0d0d",
    "bg_sidebar": "#1a1a1a",
    "bg_card": "#242424",
    "bg_hover": "#333333",
    "bg_input": "#1e1e10",
    "bg_console": "#111111",
    "bg_chart": "#0a0a0a",
    "gold": "#FFD700",
    "gold_light": "#FFE55C",
    "gold_dark": "#C9A800",
    "text_primary": "#87CEFA",
    "text_secondary": "#B0B0B0",
    "text_muted": "#808080",
    "success": "#4CAF50",
    "error": "#F44336",
    "warning": "#FF9800",
    "info": "#2196F3",
    "border": "#3a3a3a",
}

FONTS = {
    "family": "Segoe UI",
    "title": ("Segoe UI", 12, "bold"),
    "header": ("Segoe UI", 10, "bold"),
    "normal": ("Segoe UI", 8),
    "small": ("Segoe UI", 7),
    "monospace": ("Consolas", 8),
    "metrics": ("Consolas", 8),
    "sidebar": ("Segoe UI", 9, "bold"),
}

# --------------------------------------------------------------------------- #
# Global state
# --------------------------------------------------------------------------- #
STATE = {
    "models": {},
    "training_in_progress": False,
    "live_running": False,
    "live_thread": None,
    "live_stop_flag": threading.Event(),
    "live_history": [],
    "feature_cols": TRAINING_FEATURES,
    "model_manager": None,
    "optimal_threshold": DEFAULT_THRESHOLD,
    "ensemble_weight": 0.5,
    "probability_multiplier": DEFAULT_PROBABILITY_MULTIPLIER,
}
STATE_LOCK = threading.Lock()


def to_py(o):
    if isinstance(o, dict):
        return {k: to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return to_py(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def detect_encoding(file_path):
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            if encoding is None:
                encoding = 'utf-8'
            if raw_data.startswith(b'\xff\xfe'):
                return 'utf-16-le'
            elif raw_data.startswith(b'\xfe\xff'):
                return 'utf-16-be'
            elif raw_data.startswith(b'\xef\xbb\xbf'):
                return 'utf-8-sig'
            return encoding
    except:
        return 'utf-8'


def curve_downsample(x, y, n=60):
    if len(x) <= n:
        return list(x), list(y)
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return list(np.array(x)[idx]), list(np.array(y)[idx])


# --------------------------------------------------------------------------- #
# Model Manager with Detailed Metrics
# --------------------------------------------------------------------------- #
class ModelManager:
    def __init__(self):
        self.xgb_model = None
        self.cat_model = None
        self.feature_cols = TRAINING_FEATURES
        self.is_trained = False
        self.training_stats = {}
        self.ensemble_weight = 0.5
        self.feature_importance = {}
        self.optimal_threshold = DEFAULT_THRESHOLD

    def train(self, csv_path, gui_callback):
        try:
            gui_callback("progress", {"stage": "load", "pct": 5, "message": "Loading training data..."})

            encoding = detect_encoding(csv_path)
            
            # Read CSV with error handling for inconsistent rows
            try:
                df = pd.read_csv(csv_path, encoding=encoding, on_bad_lines='skip')
            except:
                # If on_bad_lines='skip' doesn't work, try more lenient approach
                df = pd.read_csv(csv_path, encoding=encoding, error_bad_lines=False, warn_bad_lines=False)
            
            df.columns = [str(c).strip() for c in df.columns]

            missing = [c for c in self.feature_cols + [TARGET_COLUMN] if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns: {missing}")

            df = df.drop_duplicates()
            df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
            df = df[df[TARGET_COLUMN].isin([0, 1])].copy()
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

            X = df[self.feature_cols].fillna(df[self.feature_cols].median())
            y = df[TARGET_COLUMN].values

            gui_callback("progress", {"stage": "split", "pct": 10, "message": "Splitting data (70/15/15)..."})

            # 70% train, 15% val, 15% test
            n = len(X)
            train_end = int(n * 0.70)
            val_end = int(n * 0.85)

            X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
            y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

            # Diagnostic: label balance per split
            rate_train, rate_val, rate_test = y_train.mean(), y_val.mean(), y_test.mean()
            max_rate_gap = max(abs(rate_train - rate_val), abs(rate_train - rate_test), abs(rate_val - rate_test))

            # Walk-forward cross-validation on the training data
            gui_callback("progress", {"stage": "cv", "pct": 20, "message": "Running walk-forward validation..."})
            wf_aucs = self._walk_forward_validate(X_train, y_train, n_splits=5)
            if wf_aucs:
                wf_mean, wf_std = float(np.mean(wf_aucs)), float(np.std(wf_aucs))
            else:
                wf_mean, wf_std = float('nan'), float('nan')

            # Baseline: plain logistic regression
            gui_callback("progress", {"stage": "baseline", "pct": 25, "message": "Fitting baseline model..."})
            baseline_auc = self._fit_baseline(X_train, y_train, X_test, y_test)

            gui_callback("progress", {"stage": "train", "pct": 35, "message": "Training XGBoost..."})

            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / max(pos_count, 1)

            self.xgb_model = xgb.XGBClassifier(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                min_child_weight=3, gamma=0.2, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.3, reg_lambda=1.5, scale_pos_weight=scale_pos_weight,
                objective='binary:logistic', eval_metric='auc',
                early_stopping_rounds=50, random_state=42, n_jobs=-1,
            )
            self.xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            gui_callback("progress", {"stage": "train", "pct": 55, "message": "Training CatBoost..."})

            self.cat_model = CatBoostClassifier(
                iterations=500, depth=6, learning_rate=0.05,
                l2_leaf_reg=3.0, loss_function='Logloss', eval_metric='AUC',
                random_seed=42, verbose=False, auto_class_weights='Balanced',
                early_stopping_rounds=50,
            )
            self.cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

            gui_callback("progress", {"stage": "validate", "pct": 75, "message": "Finding optimal ensemble..."})

            xgb_val_prob = self.xgb_model.predict_proba(X_val)[:, 1]
            cat_val_prob = self.cat_model.predict_proba(X_val)[:, 1]

            best_auc = 0
            best_weight = 0.5
            for w in np.arange(0, 1.1, 0.1):
                blend = w * xgb_val_prob + (1 - w) * cat_val_prob
                auc = roc_auc_score(y_val, blend)
                if auc > best_auc:
                    best_auc = auc
                    best_weight = w

            self.ensemble_weight = best_weight
            STATE["ensemble_weight"] = best_weight

            # Feature importance
            self.feature_importance = {
                'xgb': dict(zip(self.feature_cols, self.xgb_model.feature_importances_)),
                'cat': dict(zip(self.feature_cols, self.cat_model.feature_importances_)),
            }

            # Find optimal threshold on validation
            ensemble_val_prob = best_weight * xgb_val_prob + (1 - best_weight) * cat_val_prob
            threshold_diag = self._find_optimal_threshold(y_val, ensemble_val_prob, gui_callback)

            # Calculate detailed test metrics
            gui_callback("progress", {"stage": "test", "pct": 90, "message": "Calculating detailed metrics..."})

            self.training_stats = self._calculate_detailed_metrics(X_test, y_test)
            self.training_stats['baseline_auc'] = baseline_auc
            self.training_stats['walkforward_auc_mean'] = wf_mean
            self.training_stats['walkforward_auc_std'] = wf_std
            self.training_stats['walkforward_folds'] = wf_aucs
            self.training_stats['positive_rate_train'] = float(rate_train)
            self.training_stats['positive_rate_val'] = float(rate_val)
            self.training_stats['positive_rate_test'] = float(rate_test)
            self.training_stats['threshold_diagnostic'] = threshold_diag

            self.is_trained = True

            gui_callback("progress", {"stage": "done", "pct": 100, "message": "Training complete!"})
            gui_callback("complete", {"results": self.training_stats, "feature_importance": self.feature_importance})

            return self.training_stats

        except Exception as e:
            import traceback
            traceback.print_exc()
            gui_callback("error", {"message": str(e)})
            raise

    def _walk_forward_validate(self, X_train, y_train, n_splits=5):
        """Expanding-window walk-forward CV on the training portion only."""
        if len(X_train) < (n_splits + 1) * 20:
            return []

        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_aucs = []
        for fold_train_idx, fold_test_idx in tscv.split(X_train):
            y_fold_train = y_train[fold_train_idx]
            y_fold_test = y_train[fold_test_idx]
            if len(np.unique(y_fold_train)) < 2 or len(np.unique(y_fold_test)) < 2:
                continue

            X_fold_train = X_train.iloc[fold_train_idx]
            X_fold_test = X_train.iloc[fold_test_idx]

            neg = (y_fold_train == 0).sum()
            pos = (y_fold_train == 1).sum()
            spw = neg / max(pos, 1)

            model = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.08,
                subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                objective='binary:logistic', eval_metric='auc',
                random_state=42, n_jobs=-1,
            )
            model.fit(X_fold_train, y_fold_train, verbose=False)
            prob = model.predict_proba(X_fold_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_fold_test, prob))

        return fold_aucs

    def _fit_baseline(self, X_train, y_train, X_test, y_test):
        """Simple linear baseline for sanity-checking the boosted ensemble."""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf.fit(X_train_scaled, y_train)
        prob = clf.predict_proba(X_test_scaled)[:, 1]
        return float(roc_auc_score(y_test, prob))

    def _find_optimal_threshold(self, y_val, y_prob, gui_callback=None):
        """Find a threshold that gives a genuine, non-degenerate filter."""
        thresholds = np.arange(0.30, 0.85, 0.01)
        best_mcc = -1.0
        best_threshold = None
        best_pred_rate = None

        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            pred_rate = y_pred.mean()
            if pred_rate < 0.15 or pred_rate > 0.85:
                continue
            mcc = matthews_corrcoef(y_val, y_pred)
            if mcc > best_mcc:
                best_mcc = mcc
                best_threshold = threshold
                best_pred_rate = pred_rate

        diagnostic = {}
        if best_threshold is None or best_mcc < MIN_MEANINGFUL_MCC:
            fallback = float(np.median(y_prob))
            self.optimal_threshold = fallback
            STATE["optimal_threshold"] = fallback
            diagnostic = {
                "reliable": False,
                "best_mcc": float(best_mcc) if best_threshold is not None else None,
                "reason": "No threshold in [0.30, 0.85] achieved MCC >= "
                          f"{MIN_MEANINGFUL_MCC} while keeping predicted-positive "
                          "rate between 15% and 85%. Falling back to the median "
                          "predicted probability rather than collapsing to "
                          "'approve nearly everything'.",
            }
        else:
            self.optimal_threshold = best_threshold
            STATE["optimal_threshold"] = best_threshold
            diagnostic = {
                "reliable": True,
                "best_mcc": float(best_mcc),
                "predicted_positive_rate": float(best_pred_rate),
            }

        return diagnostic

    def _calculate_detailed_metrics(self, X_test, y_test):
        """Calculate comprehensive metrics on test set."""
        metrics = {}

        # Model predictions
        xgb_prob = self.xgb_model.predict_proba(X_test)[:, 1]
        cat_prob = self.cat_model.predict_proba(X_test)[:, 1]
        ensemble_prob = self.ensemble_weight * xgb_prob + (1 - self.ensemble_weight) * cat_prob

        # AUC scores
        metrics['xgb_auc'] = roc_auc_score(y_test, xgb_prob)
        metrics['cat_auc'] = roc_auc_score(y_test, cat_prob)
        metrics['ensemble_auc'] = roc_auc_score(y_test, ensemble_prob)
        metrics['avg_precision'] = average_precision_score(y_test, ensemble_prob)

        # Predictions at optimal threshold
        y_pred = (ensemble_prob >= self.optimal_threshold).astype(int)

        # Classification metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
        metrics['log_loss'] = log_loss(y_test, ensemble_prob, labels=[0, 1])
        metrics['brier_score'] = brier_score_loss(y_test, ensemble_prob)

        # Baseline comparisons
        base_rate = y_test.mean()
        baseline_pred = np.full_like(ensemble_prob, fill_value=base_rate)
        metrics['baseline_log_loss'] = log_loss(y_test, baseline_pred, labels=[0, 1])
        metrics['predicted_positive_rate'] = float(y_pred.mean())

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp'] = cm.ravel()

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, ensemble_prob)
        metrics['roc_fpr'], metrics['roc_tpr'] = curve_downsample(fpr, tpr)

        # Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, ensemble_prob)
        metrics['pr_precision'], metrics['pr_recall'] = curve_downsample(precision_curve, recall_curve)

        # Data info
        metrics['n_test'] = len(y_test)
        metrics['positive_rate'] = y_test.mean()
        metrics['ensemble_weight'] = self.ensemble_weight
        metrics['optimal_threshold'] = self.optimal_threshold
        metrics['feature_count'] = len(self.feature_cols)

        # Feature importance (top 20)
        xgb_imp = sorted(self.feature_importance['xgb'].items(), key=lambda x: x[1], reverse=True)[:20]
        cat_imp = sorted(self.feature_importance['cat'].items(), key=lambda x: x[1], reverse=True)[:20]
        metrics['top_features_xgb'] = xgb_imp
        metrics['top_features_cat'] = cat_imp

        return metrics

    def predict(self, features):
        if not self.is_trained:
            raise ValueError("Models not trained")

        X = pd.DataFrame([features])
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[self.feature_cols].fillna(0.0)

        xgb_prob = float(self.xgb_model.predict_proba(X)[:, 1][0])
        cat_prob = float(self.cat_model.predict_proba(X)[:, 1][0])
        ensemble_prob = self.ensemble_weight * xgb_prob + (1 - self.ensemble_weight) * cat_prob
        
        # Apply probability multiplier from settings
        multiplier = STATE.get("probability_multiplier", DEFAULT_PROBABILITY_MULTIPLIER)
        ensemble_prob_multiplied = min(ensemble_prob * multiplier, 0.99)
        xgb_prob_multiplied = min(xgb_prob * multiplier, 0.99)
        cat_prob_multiplied = min(cat_prob * multiplier, 0.99)

        return {
            'xgb_probability': xgb_prob_multiplied,
            'cat_probability': cat_prob_multiplied,
            'ensemble_probability': ensemble_prob_multiplied,
            'raw_ensemble_probability': ensemble_prob,
        }

    def save_models(self, filename="signal_filter_models.joblib"):
        if not self.is_trained:
            return
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump({
            'xgb_model': self.xgb_model,
            'cat_model': self.cat_model,
            'feature_cols': self.feature_cols,
            'training_stats': self.training_stats,
            'ensemble_weight': self.ensemble_weight,
            'feature_importance': self.feature_importance,
            'optimal_threshold': self.optimal_threshold,
        }, path)
        return path

    def load_models(self, filename="signal_filter_models.joblib"):
        path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(path):
            data = joblib.load(path)
            self.xgb_model = data['xgb_model']
            self.cat_model = data['cat_model']
            self.feature_cols = data['feature_cols']
            self.training_stats = data.get('training_stats', {})
            self.ensemble_weight = data.get('ensemble_weight', 0.5)
            self.feature_importance = data.get('feature_importance', {})
            self.optimal_threshold = data.get('optimal_threshold', DEFAULT_THRESHOLD)
            self.is_trained = True
            return True
        return False


# --------------------------------------------------------------------------- #
# Live signal functions (same as before)
# --------------------------------------------------------------------------- #
def parse_live_signal(path):
    if not os.path.exists(path):
        return None

    encoding = detect_encoding(path)
    try:
        with open(path, "r", encoding=encoding, newline="") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
    except:
        with open(path, "r", encoding="utf-8", newline="") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    if len(lines) < 2:
        return None

    header = lines[0].split(',')
    data = lines[-1].split(',')

    if len(data) < len(header):
        return None

    rec = {}
    for i, col in enumerate(header):
        if i < len(data):
            rec[col.strip()] = data[i].strip()

    try:
        rec["signal_id"] = int(float(rec.get("signal_id", 0)))
        rec["entry_price"] = float(rec.get("entry_price", 0))
        rec["sl_price"] = float(rec.get("sl_price", 0))
        rec["tp1_price"] = float(rec.get("tp1_price", 0))
        rec["tp2_price"] = float(rec.get("tp2_price", 0))
        rec["tp3_price"] = float(rec.get("tp3_price", 0))
    except:
        pass

    for col in TRAINING_FEATURES:
        if col in rec:
            try:
                rec[col] = float(rec[col])
            except:
                rec[col] = 0.0

    return rec


def recommendation_for(prob, threshold):
    if prob >= threshold + 0.10:
        return "STRONG_TAKE"
    elif prob >= threshold:
        return "TAKE"
    elif prob >= threshold - 0.05:
        return "CAUTION"
    else:
        return "SKIP"


def write_ea_prediction(folder, rec, probability, recommendation):
    """
    Write the EA prediction output CSV with a header row and full signal
    details: signal_id, timestamp, symbol, direction, entry, sl, tp1, tp2,
    tp3, probability, recommendation.
    """
    path = os.path.join(folder, EA_PREDICTION_FILE)
    row = [
        rec.get("signal_id", ""),
        rec.get("timestamp", ""),
        rec.get("symbol", ""),
        rec.get("direction", ""),
        rec.get("entry_price", ""),
        rec.get("sl_price", ""),
        rec.get("tp1_price", ""),
        rec.get("tp2_price", ""),
        rec.get("tp3_price", ""),
        f"{probability:.4f}",
        recommendation,
    ]
    file_exists = os.path.exists(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(EA_PREDICTION_HEADER)
        writer.writerow(row)


def live_loop(folder, threshold, poll_interval, gui_callback):
    stop_flag = STATE["live_stop_flag"]
    signal_path = os.path.join(folder, LIVE_SIGNAL_FILE)
    last_id = 0

    gui_callback("live_status", {"running": True})

    while not stop_flag.is_set():
        try:
            rec = parse_live_signal(signal_path)

            if rec and rec["signal_id"] != last_id:
                last_id = rec["signal_id"]

                model_manager = STATE.get("model_manager")

                if model_manager and model_manager.is_trained:
                    prediction = model_manager.predict(rec)
                    recommendation = recommendation_for(prediction['ensemble_probability'], threshold)

                    write_ea_prediction(folder, rec, prediction['ensemble_probability'], recommendation)

                    payload = to_py({
                        "signal_id": rec["signal_id"],
                        "timestamp": rec.get("timestamp", ""),
                        "symbol": rec.get("symbol", ""),
                        "direction": rec.get("direction", ""),
                        "entry": rec.get("entry_price", 0),
                        "sl": rec.get("sl_price", 0),
                        "tp1": rec.get("tp1_price", 0),
                        "tp2": rec.get("tp2_price", 0),
                        "tp3": rec.get("tp3_price", 0),
                        "probability": round(prediction['ensemble_probability'], 4),
                        "xgb_prob": round(prediction['xgb_probability'], 4),
                        "cat_prob": round(prediction['cat_probability'], 4),
                        "recommendation": recommendation,
                        "received_at": datetime.now().strftime("%H:%M:%S"),
                    })

                    with STATE_LOCK:
                        STATE["live_history"].insert(0, payload)
                        STATE["live_history"] = STATE["live_history"][:200]

                    gui_callback("live_signal", payload)

        except Exception as e:
            pass

        stop_flag.wait(poll_interval)

    gui_callback("live_status", {"running": False})


# --------------------------------------------------------------------------- #
# GUI Application
# --------------------------------------------------------------------------- #
class QueenFlowApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Flow ML - Signal Filter with Detailed Metrics")
        self.root.geometry("1400x520")
        self.root.configure(bg=COLORS["bg_dark"])

        self.model_manager = ModelManager()
        STATE["model_manager"] = self.model_manager

        if self.model_manager.load_models():
            print("Loaded existing models")

        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('.', background=COLORS["bg_dark"], foreground=COLORS["text_primary"], font=FONTS["normal"])
        self.style.configure('TFrame', background=COLORS["bg_dark"])
        self.style.configure('TLabel', background=COLORS["bg_dark"], foreground=COLORS["text_primary"], font=FONTS["normal"])
        self.style.configure('Sidebar.TFrame', background=COLORS["bg_sidebar"])
        self.style.configure('Card.TFrame', background=COLORS["bg_card"])
        self.style.configure('Card.TLabel', background=COLORS["bg_card"], foreground=COLORS["text_primary"], font=FONTS["normal"])
        self.style.configure('Header.TLabel', background=COLORS["bg_dark"], foreground=COLORS["gold"], font=FONTS["title"])
        self.style.configure('Accent.TButton', background=COLORS["gold"], foreground=COLORS["bg_dark"], font=FONTS["header"], padding=[20, 8])
        self.style.map('Accent.TButton', background=[('active', COLORS["gold_light"])])
        self.style.configure('TButton', font=FONTS["normal"], padding=[10, 5])
        self.style.configure('TEntry', fieldbackground=COLORS["bg_input"], foreground=COLORS["text_primary"], insertcolor=COLORS["gold"], font=FONTS["normal"])
        self.style.configure('TCombobox', fieldbackground=COLORS["bg_input"], foreground=COLORS["text_primary"], font=FONTS["normal"], arrowcolor=COLORS["gold"])
        self.style.configure('Horizontal.TProgressbar', background=COLORS["text_primary"], troughcolor=COLORS["bg_input"])
        self.style.configure('Treeview', background=COLORS["bg_card"], foreground=COLORS["text_primary"], fieldbackground=COLORS["bg_card"], font=FONTS["small"], rowheight=18)
        self.style.configure('Treeview.Heading', background=COLORS["bg_hover"], foreground=COLORS["gold"], font=FONTS["normal"])
        self.style.configure('TNotebook', background=COLORS["bg_dark"], borderwidth=0)
        self.style.configure('TNotebook.Tab', font=FONTS["normal"], padding=[10, 4])

        self.setup_layout()
        self.gui_queue = queue.Queue()
        self.root.after(100, self.process_queue)

    def setup_layout(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True)

        self.sidebar = ttk.Frame(self.main_container, style='Sidebar.TFrame', width=200)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)

        logo_frame = tk.Frame(self.sidebar, bg=COLORS["bg_sidebar"])
        logo_frame.pack(fill=tk.X, pady=20)
        tk.Label(logo_frame, text="QF", bg=COLORS["gold"], fg=COLORS["bg_dark"],
                font=("Segoe UI", 18, "bold"), width=3, height=1).pack(pady=5)
        tk.Label(logo_frame, text="Flow ML", bg=COLORS["bg_sidebar"],
                fg=COLORS["gold"], font=FONTS["title"]).pack()

        ttk.Separator(self.sidebar, orient='horizontal').pack(fill=tk.X, pady=10)

        self.nav_buttons = {}
        nav_items = [
            ("training", "📊 Training", self.show_training),
            ("live", "🔴 Live Trading", self.show_live),
            ("models", "📦 Models", self.show_models),
            ("settings", "⚙️ Settings", self.show_settings),
        ]

        for key, text, command in nav_items:
            btn = tk.Button(
                self.sidebar, text=text, command=command,
                bg=COLORS["bg_sidebar"], fg=COLORS["text_secondary"],
                font=FONTS["sidebar"], bd=0, padx=15, pady=10,
                anchor='w', cursor='hand2', activebackground=COLORS["bg_hover"],
                activeforeground=COLORS["gold"]
            )
            btn.pack(fill=tk.X, pady=2)
            self.nav_buttons[key] = btn

        ttk.Separator(self.sidebar, orient='horizontal').pack(fill=tk.X, pady=10)

        status_frame = tk.Frame(self.sidebar, bg=COLORS["bg_sidebar"])
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        self.status_dot = tk.Label(status_frame, text="●", bg=COLORS["bg_sidebar"],
                                   fg=COLORS["success"], font=("Segoe UI", 10))
        self.status_dot.pack(side=tk.LEFT, padx=5)
        tk.Label(status_frame, text="Ready", bg=COLORS["bg_sidebar"],
                fg=COLORS["text_secondary"], font=FONTS["small"]).pack(side=tk.LEFT)

        self.content_frame = ttk.Frame(self.main_container)
        self.content_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.pages = {}
        self.create_training_page()
        self.create_live_page()
        self.create_models_page()
        self.create_settings_page()

        self.show_training()

    def create_training_page(self):
        page = ttk.Frame(self.content_frame)
        self.pages["training"] = page

        header = ttk.Frame(page)
        header.pack(fill=tk.X, padx=20, pady=(15, 10))
        ttk.Label(header, text="Model Training - Signal Filter", style='Header.TLabel').pack(anchor=tk.W)

        content = ttk.Frame(page)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        left_panel = ttk.Frame(content, style='Card.TFrame', padding=15)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(left_panel, text="Training CSV", style='Card.TLabel').pack(anchor=tk.W, pady=(5, 2))
        csv_frame = ttk.Frame(left_panel, style='Card.TFrame')
        csv_frame.pack(fill=tk.X)
        self.csv_path_var = tk.StringVar()
        ttk.Entry(csv_frame, textvariable=self.csv_path_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(csv_frame, text="Browse", command=self.browse_csv, width=8).pack(side=tk.LEFT, padx=(5, 0))

        self.train_button = ttk.Button(left_panel, text="Train Signal Filter",
                                       style='Accent.TButton', command=self.start_training)
        self.train_button.pack(fill=tk.X, pady=(20, 0))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(left_panel, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(10, 0))

        right_panel = ttk.Frame(content)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.training_notebook = ttk.Notebook(right_panel)
        self.training_notebook.pack(fill=tk.BOTH, expand=True)

        console_frame = tk.Frame(self.training_notebook, bg=COLORS["bg_console"])
        self.training_notebook.add(console_frame, text="Console")
        self.console_text = scrolledtext.ScrolledText(
            console_frame, bg=COLORS["bg_console"], fg=COLORS["text_primary"],
            insertbackground=COLORS["gold"], font=FONTS["monospace"],
            relief=tk.FLAT, borderwidth=0
        )
        self.console_text.pack(fill=tk.BOTH, expand=True)

        self.metrics_frame = tk.Frame(self.training_notebook, bg=COLORS["bg_console"])
        self.training_notebook.add(self.metrics_frame, text="Detailed Metrics")

        self.charts_frame = tk.Frame(self.training_notebook, bg=COLORS["bg_chart"])
        self.training_notebook.add(self.charts_frame, text="Charts")

    def create_live_page(self):
        page = ttk.Frame(self.content_frame)
        self.pages["live"] = page

        header = ttk.Frame(page)
        header.pack(fill=tk.X, padx=20, pady=(15, 10))
        ttk.Label(header, text="Live Trading - Signal Filtering", style='Header.TLabel').pack(anchor=tk.W)

        content = ttk.Frame(page)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)

        left_panel = ttk.Frame(content, style='Card.TFrame', padding=15)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        ttk.Label(left_panel, text="MT5 Files Folder", style='Card.TLabel').pack(anchor=tk.W, pady=(5, 2))
        folder_frame = ttk.Frame(left_panel, style='Card.TFrame')
        folder_frame.pack(fill=tk.X)
        self.folder_path_var = tk.StringVar()
        ttk.Entry(folder_frame, textvariable=self.folder_path_var, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(folder_frame, text="Browse", command=self.browse_folder, width=8).pack(side=tk.LEFT, padx=(5, 0))

        ttk.Label(left_panel, text="Model", style='Card.TLabel').pack(anchor=tk.W, pady=(15, 2))
        self.live_model_var = tk.StringVar(value="ensemble")
        ttk.Combobox(left_panel, textvariable=self.live_model_var,
                     values=["ensemble", "xgboost", "catboost"], width=28).pack(fill=tk.X)

        ttk.Label(left_panel, text="Threshold", style='Card.TLabel').pack(anchor=tk.W, pady=(15, 2))
        self.threshold_var = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        threshold_frame = ttk.Frame(left_panel, style='Card.TFrame')
        threshold_frame.pack(fill=tk.X)
        ttk.Scale(threshold_frame, from_=0.40, to=0.75, variable=self.threshold_var,
                 orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.threshold_label = ttk.Label(threshold_frame, text="0.60", style='Card.TLabel', width=5)
        self.threshold_label.pack(side=tk.LEFT, padx=(5, 0))
        self.threshold_var.trace('w', lambda *args: self.threshold_label.config(
            text=f"{self.threshold_var.get():.2f}"))

        ttk.Label(left_panel, text="Poll Interval (sec)", style='Card.TLabel').pack(anchor=tk.W, pady=(15, 2))
        self.poll_interval_var = tk.DoubleVar(value=2)
        ttk.Spinbox(left_panel, from_=1, to=30, textvariable=self.poll_interval_var, width=28).pack(fill=tk.X)

        button_frame = ttk.Frame(left_panel, style='Card.TFrame')
        button_frame.pack(fill=tk.X, pady=(20, 0))
        self.live_start_button = ttk.Button(button_frame, text="Start", style='Accent.TButton',
                                            command=self.start_live_loop)
        self.live_start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.live_stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_live_loop,
                                           state=tk.DISABLED)
        self.live_stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        right_panel = ttk.Frame(content)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        signal_card = ttk.Frame(right_panel, style='Card.TFrame', padding=15)
        signal_card.pack(fill=tk.X)
        self.signal_info_var = tk.StringVar(value="Waiting for signal...")
        tk.Label(signal_card, textvariable=self.signal_info_var, bg=COLORS["bg_card"],
                fg=COLORS["gold"], font=("Segoe UI", 11, "bold")).pack()

        price_frame = tk.Frame(signal_card, bg=COLORS["bg_card"])
        price_frame.pack(fill=tk.X, pady=(5, 0))

        self.entry_label = tk.Label(price_frame, text="Entry: --", bg=COLORS["bg_card"],
                                    fg=COLORS["text_secondary"], font=FONTS["normal"])
        self.entry_label.pack(side=tk.LEFT, padx=5)
        self.sl_label = tk.Label(price_frame, text="SL: --", bg=COLORS["bg_card"],
                                fg=COLORS["error"], font=FONTS["normal"])
        self.sl_label.pack(side=tk.LEFT, padx=5)
        self.tp1_label = tk.Label(price_frame, text="TP1: --", bg=COLORS["bg_card"],
                                 fg=COLORS["success"], font=FONTS["normal"])
        self.tp1_label.pack(side=tk.LEFT, padx=5)

        history_card = ttk.Frame(right_panel, style='Card.TFrame', padding=10)
        history_card.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        ttk.Label(history_card, text="Signal History", style='Card.TLabel',
                 font=FONTS["header"]).pack(anchor=tk.W, pady=(0, 5))

        self.signal_tree = ttk.Treeview(
            history_card,
            columns=("time", "symbol", "dir", "entry", "sl", "tp1", "tp2", "tp3", "prob", "verdict"),
            show="headings",
            height=20
        )

        column_configs = {
            "time": ("TIME", 60),
            "symbol": ("SYMBOL", 70),
            "dir": ("DIR", 50),
            "entry": ("ENTRY", 70),
            "sl": ("SL", 70),
            "tp1": ("TP1", 70),
            "tp2": ("TP2", 70),
            "tp3": ("TP3", 70),
            "prob": ("PROB", 60),
            "verdict": ("VERDICT", 100),
        }

        for col, (heading, width) in column_configs.items():
            self.signal_tree.heading(col, text=heading)
            self.signal_tree.column(col, width=width, anchor=tk.CENTER)

        tree_scroll_y = ttk.Scrollbar(history_card, orient=tk.VERTICAL, command=self.signal_tree.yview)
        tree_scroll_x = ttk.Scrollbar(history_card, orient=tk.HORIZONTAL, command=self.signal_tree.xview)
        self.signal_tree.configure(yscrollcommand=tree_scroll_y.set, xscrollcommand=tree_scroll_x.set)

        self.signal_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

    def create_models_page(self):
        page = ttk.Frame(self.content_frame)
        self.pages["models"] = page

        header = ttk.Frame(page)
        header.pack(fill=tk.X, padx=20, pady=(15, 10))
        ttk.Label(header, text="Saved Models", style='Header.TLabel').pack(anchor=tk.W)

        content = ttk.Frame(page, style='Card.TFrame', padding=20)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        if self.model_manager.is_trained:
            stats = self.model_manager.training_stats
            ttk.Label(content, text=f"Model Status: TRAINED", style='Card.TLabel',
                     font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=5)
            ttk.Label(content, text=f"XGBoost AUC: {stats.get('xgb_auc', 'N/A'):.4f}",
                     style='Card.TLabel').pack(anchor=tk.W, pady=2)
            ttk.Label(content, text=f"CatBoost AUC: {stats.get('cat_auc', 'N/A'):.4f}",
                     style='Card.TLabel').pack(anchor=tk.W, pady=2)
            ttk.Label(content, text=f"Ensemble AUC: {stats.get('ensemble_auc', 'N/A'):.4f}",
                     style='Card.TLabel').pack(anchor=tk.W, pady=2)
            ttk.Label(content, text=f"Baseline (LogReg) AUC: {stats.get('baseline_auc', float('nan')):.4f}",
                     style='Card.TLabel').pack(anchor=tk.W, pady=2)
            ttk.Label(content, text=f"Walk-forward AUC: {stats.get('walkforward_auc_mean', float('nan')):.4f} "
                                     f"+/- {stats.get('walkforward_auc_std', float('nan')):.4f}",
                     style='Card.TLabel').pack(anchor=tk.W, pady=2)
            ttk.Label(content, text=f"Test Accuracy: {stats.get('accuracy', 'N/A'):.4f}",
                     style='Card.TLabel').pack(anchor=tk.W, pady=2)
            ttk.Label(content, text=f"Test F1: {stats.get('f1', 'N/A'):.4f}",
                     style='Card.TLabel').pack(anchor=tk.W, pady=2)
            ttk.Label(content, text=f"Test MCC: {stats.get('mcc', 'N/A'):.4f}",
                     style='Card.TLabel').pack(anchor=tk.W, pady=2)
            ttk.Label(content, text=f"Optimal Threshold: {stats.get('optimal_threshold', 'N/A'):.2f}",
                     style='Card.TLabel').pack(anchor=tk.W, pady=2)
            ttk.Label(content, text=f"Test Samples: {stats.get('n_test', 'N/A')}",
                     style='Card.TLabel').pack(anchor=tk.W, pady=2)

            ttk.Button(content, text="Save Models", command=self.save_models).pack(pady=10)
        else:
            ttk.Label(content, text="No trained models. Train models first.",
                     style='Card.TLabel', font=("Segoe UI", 11)).pack(pady=20)

    def create_settings_page(self):
        page = ttk.Frame(self.content_frame)
        self.pages["settings"] = page

        header = ttk.Frame(page)
        header.pack(fill=tk.X, padx=20, pady=(15, 10))
        ttk.Label(header, text="Settings", style='Header.TLabel').pack(anchor=tk.W)

        content = ttk.Frame(page, style='Card.TFrame', padding=20)
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        ttk.Label(content, text="Default Threshold:", style='Card.TLabel').pack(anchor=tk.W, pady=5)
        self.default_threshold_var = tk.DoubleVar(value=DEFAULT_THRESHOLD)
        ttk.Scale(content, from_=0.40, to=0.75, variable=self.default_threshold_var,
                 orient=tk.HORIZONTAL).pack(fill=tk.X)

        ttk.Label(content, text="Probability Multiplier:", style='Card.TLabel').pack(anchor=tk.W, pady=(15, 5))
        self.probability_multiplier_var = tk.DoubleVar(value=DEFAULT_PROBABILITY_MULTIPLIER)
        multiplier_frame = ttk.Frame(content, style='Card.TFrame')
        multiplier_frame.pack(fill=tk.X)
        ttk.Scale(multiplier_frame, from_=0.8, to=1.5, variable=self.probability_multiplier_var,
                 orient=tk.HORIZONTAL).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.multiplier_label = ttk.Label(multiplier_frame, text="1.19", style='Card.TLabel', width=5)
        self.multiplier_label.pack(side=tk.LEFT, padx=(5, 0))
        self.probability_multiplier_var.trace('w', lambda *args: self.multiplier_label.config(
            text=f"{self.probability_multiplier_var.get():.2f}"))

        ttk.Label(content, text="Poll Interval (seconds):", style='Card.TLabel').pack(anchor=tk.W, pady=(15, 5))
        self.default_poll_var = tk.DoubleVar(value=2)
        ttk.Spinbox(content, from_=1, to=30, textvariable=self.default_poll_var, width=10).pack(anchor=tk.W)

        ttk.Button(content, text="Save Settings", command=self.save_settings).pack(pady=20)

    def show_training(self):
        self.show_page("training")

    def show_live(self):
        self.show_page("live")

    def show_models(self):
        self.show_page("models")

    def show_settings(self):
        self.show_page("settings")

    def show_page(self, page_name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[page_name].pack(fill=tk.BOTH, expand=True)
        for key, btn in self.nav_buttons.items():
            if key == page_name:
                btn.config(bg=COLORS["bg_hover"], fg=COLORS["gold"])
            else:
                btn.config(bg=COLORS["bg_sidebar"], fg=COLORS["text_secondary"])

    def browse_csv(self):
        filename = filedialog.askopenfilename(title="Select Training CSV",
                                              filetypes=[("CSV files", "*.csv")])
        if filename:
            self.csv_path_var.set(filename)

    def browse_folder(self):
        folder = filedialog.askdirectory(title="Select MT5 Files Folder")
        if folder:
            self.folder_path_var.set(folder)

    def save_models(self):
        path = self.model_manager.save_models()
        if path:
            messagebox.showinfo("Success", f"Models saved to:\n{path}")
        else:
            messagebox.showwarning("Warning", "No trained models to save")

    def save_settings(self):
        # Save probability multiplier to STATE
        STATE["probability_multiplier"] = self.probability_multiplier_var.get()
        messagebox.showinfo("Success", f"Settings saved! Probability multiplier: {STATE['probability_multiplier']:.2f}")

    def start_training(self):
        csv_path = self.csv_path_var.get().strip()
        if not csv_path:
            messagebox.showerror("Error", "Please select a training CSV")
            return
        if STATE["training_in_progress"]:
            messagebox.showwarning("Warning", "Training in progress")
            return

        self.train_button.config(state=tk.DISABLED)
        self.console_text.delete(1.0, tk.END)

        thread = threading.Thread(
            target=self.model_manager.train,
            args=(csv_path, self.gui_callback),
            daemon=True
        )
        thread.start()

    def start_live_loop(self):
        folder = self.folder_path_var.get().strip()
        if not folder:
            messagebox.showerror("Error", "Select folder")
            return
        if not self.model_manager.is_trained:
            messagebox.showerror("Error", "Train models first")
            return

        with STATE_LOCK:
            if STATE["live_running"]:
                return
            STATE["live_running"] = True
            STATE["live_stop_flag"].clear()

        self.live_start_button.config(state=tk.DISABLED)
        self.live_stop_button.config(state=tk.NORMAL)

        thread = threading.Thread(
            target=live_loop,
            args=(folder, self.threshold_var.get(),
                  self.poll_interval_var.get(), self.gui_callback),
            daemon=True
        )
        thread.start()

    def stop_live_loop(self):
        with STATE_LOCK:
            if not STATE["live_running"]:
                return
            STATE["live_stop_flag"].set()
            STATE["live_running"] = False

        self.live_start_button.config(state=tk.NORMAL)
        self.live_stop_button.config(state=tk.DISABLED)

    def gui_callback(self, event_type, data):
        self.gui_queue.put((event_type, data))

    def process_queue(self):
        try:
            while True:
                event_type, data = self.gui_queue.get_nowait()

                if event_type == "log":
                    self.log_message(data["message"], data.get("level", "info"))
                elif event_type == "progress":
                    self.update_progress(data["pct"], data["message"])
                elif event_type == "complete":
                    self.training_complete(data)
                elif event_type == "error":
                    self.training_error(data)
                elif event_type == "live_signal":
                    self.update_live_signal(data)
                elif event_type == "live_status":
                    self.update_live_status(data)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self.process_queue)

    def log_message(self, message, level="info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = {"info": "[INFO]", "success": "[OK]", "error": "[ERR]", "warning": "[WARN]"}.get(level, "[INFO]")
        self.console_text.insert(tk.END, f"{timestamp} {prefix} {message}\n")
        self.console_text.see(tk.END)
        self.status_dot.config(fg=COLORS["success"] if level != "error" else COLORS["error"])

    def update_progress(self, pct, message):
        self.progress_var.set(pct)

    def training_complete(self, data):
        self.train_button.config(state=tk.NORMAL)
        self.progress_var.set(100)
        self.display_results(data["results"])

    def training_error(self, data):
        self.train_button.config(state=tk.NORMAL)

    def update_live_signal(self, signal):
        info_text = f"#{signal['signal_id']} - {signal['symbol']} {signal['direction']} "
        info_text += f"Prob: {signal['probability']*100:.1f}% ({signal['recommendation']})"
        self.signal_info_var.set(info_text)

        self.entry_label.config(text=f"Entry: {signal['entry']:.5f}")
        self.sl_label.config(text=f"SL: {signal['sl']:.5f}")
        self.tp1_label.config(text=f"TP1: {signal['tp1']:.5f}")

        self.signal_tree.insert("", 0, values=(
            signal['received_at'],
            signal['symbol'],
            signal['direction'],
            f"{signal['entry']:.5f}",
            f"{signal['sl']:.5f}",
            f"{signal['tp1']:.5f}",
            f"{signal['tp2']:.5f}",
            f"{signal['tp3']:.5f}",
            f"{signal['probability']*100:.1f}%",
            signal['recommendation']
        ))

    def update_live_status(self, data):
        if data["running"]:
            self.live_start_button.config(state=tk.DISABLED)
            self.live_stop_button.config(state=tk.NORMAL)
            self.status_dot.config(fg=COLORS["warning"])
        else:
            self.live_start_button.config(state=tk.NORMAL)
            self.live_stop_button.config(state=tk.DISABLED)
            self.status_dot.config(fg=COLORS["success"])

    def display_results(self, results):
        for widget in self.metrics_frame.winfo_children():
            widget.destroy()

        metrics_text = tk.Text(
            self.metrics_frame, bg=COLORS["bg_console"], fg=COLORS["text_primary"],
            font=FONTS["metrics"], relief=tk.FLAT, borderwidth=0,
            padx=15, pady=10
        )
        metrics_text.pack(fill=tk.BOTH, expand=True)

        text = "=" * 70 + "\n"
        text += "SIGNAL FILTER - DETAILED PERFORMANCE REPORT\n"
        text += "=" * 70 + "\n\n"

        # Model AUCs
        text += "MODEL PERFORMANCE (AUC):\n"
        text += f"  XGBoost:            {results.get('xgb_auc', 0):.4f}\n"
        text += f"  CatBoost:           {results.get('cat_auc', 0):.4f}\n"
        text += f"  ENSEMBLE:           {results.get('ensemble_auc', 0):.4f}\n"
        text += f"  Baseline (LogReg):  {results.get('baseline_auc', 0):.4f}\n"
        wf_mean = results.get('walkforward_auc_mean', float('nan'))
        wf_std = results.get('walkforward_auc_std', float('nan'))
        text += f"  Walk-forward:       {wf_mean:.4f} +/- {wf_std:.4f}\n\n"

        # Classification metrics
        text += "CLASSIFICATION METRICS (at optimal threshold):\n"
        text += f"  Optimal Threshold: {results.get('optimal_threshold', 0):.2f}\n"
        text += f"  Accuracy:          {results.get('accuracy', 0):.4f}\n"
        text += f"  Precision:         {results.get('precision', 0):.4f}\n"
        text += f"  Recall:            {results.get('recall', 0):.4f}\n"
        text += f"  F1 Score:          {results.get('f1', 0):.4f}\n"
        text += f"  MCC:               {results.get('mcc', 0):.4f}\n"
        text += f"  Avg Precision:     {results.get('avg_precision', 0):.4f}\n"
        text += f"  Log Loss:          {results.get('log_loss', 0):.4f} (baseline: {results.get('baseline_log_loss', 0):.4f})\n"
        text += f"  Brier Score:       {results.get('brier_score', 0):.4f}\n"
        text += f"  Predicted Positive Rate: {results.get('predicted_positive_rate', 0):.1%}\n\n"

        # Confusion matrix
        text += "CONFUSION MATRIX:\n"
        text += f"  TN={results.get('tn', 0):>5}  FP={results.get('fp', 0):>5}\n"
        text += f"  FN={results.get('fn', 0):>5}  TP={results.get('tp', 0):>5}\n\n"

        # Threshold diagnostic
        thr_diag = results.get('threshold_diagnostic', {})
        text += "THRESHOLD SELECTION:\n"
        if thr_diag.get('reliable'):
            text += f"  Reliable non-degenerate threshold found. Val MCC: {thr_diag.get('best_mcc', 0):.4f}, " \
                    f"predicted positive rate: {thr_diag.get('predicted_positive_rate', 0):.1%}\n\n"
        else:
            text += f"  UNRELIABLE - {thr_diag.get('reason', '')}\n\n"

        # Data info
        text += "DATA INFORMATION:\n"
        text += f"  Test Samples:    {results.get('n_test', 0)}\n"
        text += f"  Positive Rate:   train={results.get('positive_rate_train', 0):.1%} " \
                f"val={results.get('positive_rate_val', 0):.1%} test={results.get('positive_rate_test', 0):.1%}\n"
        text += f"  Total Features:  {results.get('feature_count', 0)}\n"
        text += f"  XGB Weight:      {results.get('ensemble_weight', 0):.2f}\n\n"

        # Top features
        if 'top_features_xgb' in results:
            text += "TOP 10 XGBOOST FEATURES:\n"
            for feat, imp in results['top_features_xgb'][:10]:
                text += f"  {feat:30s}: {imp:.4f}\n"
            text += "\n"

        if 'top_features_cat' in results:
            text += "TOP 10 CATBOOST FEATURES:\n"
            for feat, imp in results['top_features_cat'][:10]:
                text += f"  {feat:30s}: {imp:.4f}\n"
            text += "\n"

        text += "=" * 70 + "\n"

        metrics_text.insert(1.0, text)
        metrics_text.config(state=tk.DISABLED)

        for widget in self.charts_frame.winfo_children():
            widget.destroy()

        self.create_charts(results)
        self.training_notebook.select(1)

    def create_charts(self, results):
        plt.style.use('dark_background')

        fig = Figure(figsize=(10, 4), dpi=90, facecolor=COLORS["bg_chart"])

        # ROC Curve
        ax1 = fig.add_subplot(131, facecolor=COLORS["bg_chart"])
        if 'roc_fpr' in results and 'roc_tpr' in results:
            ax1.plot(results['roc_fpr'], results['roc_tpr'], color='#FFD700', linewidth=2)
            ax1.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
            ax1.set_xlabel("FPR", fontsize=7, color=COLORS["text_primary"])
            ax1.set_ylabel("TPR", fontsize=7, color=COLORS["text_primary"])
            ax1.set_title(f"ROC (AUC={results.get('ensemble_auc', 0):.3f})",
                         color=COLORS["gold"], fontsize=8)
            ax1.tick_params(labelsize=6, colors=COLORS["text_secondary"])
            ax1.grid(True, alpha=0.2)

        # Model Comparison
        ax2 = fig.add_subplot(132, facecolor=COLORS["bg_chart"])
        models = ['XGB', 'CAT', 'ENS', 'Base', 'WF']
        aucs = [
            results.get('xgb_auc', 0),
            results.get('cat_auc', 0),
            results.get('ensemble_auc', 0),
            results.get('baseline_auc', 0),
            results.get('walkforward_auc_mean', 0) or 0,
        ]
        colors = ['#FFD700', '#87CEFA', '#4CAF50', '#B0B0B0', '#FF9800']
        ax2.bar(models, aucs, color=colors, alpha=0.8)
        ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1)
        ax2.set_ylim(0.4, max(0.7, max(aucs) + 0.05))
        ax2.set_title('AUC Comparison', color=COLORS["gold"], fontsize=8)
        ax2.tick_params(labelsize=6, colors=COLORS["text_secondary"])
        ax2.grid(True, alpha=0.2, axis='y')

        # Precision-Recall
        ax3 = fig.add_subplot(133, facecolor=COLORS["bg_chart"])
        if 'pr_precision' in results and 'pr_recall' in results:
            ax3.plot(results['pr_recall'], results['pr_precision'], color='#4CAF50', linewidth=2)
            ax3.set_xlabel("Recall", fontsize=7, color=COLORS["text_primary"])
            ax3.set_ylabel("Precision", fontsize=7, color=COLORS["text_primary"])
            ax3.set_title("Precision-Recall", color=COLORS["gold"], fontsize=8)
            ax3.tick_params(labelsize=6, colors=COLORS["text_secondary"])
            ax3.grid(True, alpha=0.2)

        for ax in [ax1, ax2, ax3]:
            for spine in ax.spines.values():
                spine.set_color(COLORS["border"])

        canvas = FigureCanvasTkAgg(fig, self.charts_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def main():
    root = tk.Tk()
    app = QueenFlowApp(root)
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    root.mainloop()


if __name__ == "__main__":
    print("=" * 70)
    print(" Flow ML")
    print("=" * 70)
    print("Starting application...")
    main()