"""
Model training, evaluation, and experiment tracking pipeline for Dynamic Pricing Engine.
"""
import sys
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Append project root to path to import config
sys.path.append(str(Path(__file__).parent.parent))
import config

def load_processed_data():
    """
    Load train and test parquet files from config paths.
    Print shapes and return train_df, test_df.
    """
    print("Loading processed data...")
    train_df = pd.read_parquet(config.PROCESSED_TRAIN)
    test_df = pd.read_parquet(config.PROCESSED_TEST)
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    return train_df, test_df

def get_feature_target(df):
    """
    Split df into X (ALL_FEATURES from config) and y (LOG_TARGET from config).
    Return X, y.
    """
    X = df[config.ALL_FEATURES]
    y = df[config.LOG_TARGET]
    return X, y

def train_baseline(X_train, y_train):
    """
    Fit a DummyRegressor(strategy="mean") on training data.
    Return fitted model.
    """
    print("\nTraining baseline model (DummyRegressor)...")
    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_train, y_train)
    return baseline_model

def train_lightgbm(X_train, y_train):
    """
    Train LightGBM with Optuna hyperparameter tuning.
    50 trials, 5-fold CV, optimize RMSE.
    Hyperparameters to tune: num_leaves (20-300), learning_rate (0.01-0.3), n_estimators (100-1000), 
    min_child_samples (5-100), subsample (0.5-1.0), colsample_bytree (0.5-1.0).
    Use optuna.logging.set_verbosity(optuna.logging.WARNING) to suppress output.
    Print best params and best CV RMSE.
    Refit final model on full training set with best params.
    Return fitted LightGBM model, best_params dict.
    """
    print("\nStarting Optuna hyperparameter tuning for LightGBM...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': config.RANDOM_SEED,
            'n_jobs': -1,
            'verbose': -1
        }
        
        cv = KFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
        cv_scores = []
        
        # Reset index to avoid alignment issues during indexing
        X_tmp = X_train.reset_index(drop=True)
        y_tmp = y_train.reset_index(drop=True)
        
        for train_idx, val_idx in cv.split(X_tmp):
            X_tr, X_val = X_tmp.iloc[train_idx], X_tmp.iloc[val_idx]
            y_tr, y_val = y_tmp.iloc[train_idx], y_tmp.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_val)
            rmse = mean_squared_error(y_val, preds, squared=False)
            cv_scores.append(rmse)
            
        return np.mean(cv_scores)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_SEED))
    study.optimize(objective, n_trials=config.N_OPTUNA_TRIALS)
    
    best_params = study.best_params
    best_rmse = study.best_value
    print(f"Best CV RMSE (log space): {best_rmse:.4f}")
    print(f"Best params: {best_params}")
    
    print("Refitting LightGBM model on full training set with best params...")
    best_params['random_state'] = config.RANDOM_SEED
    best_params['n_jobs'] = -1
    best_params['verbose'] = -1
    
    final_model = lgb.LGBMRegressor(**best_params)
    final_model.fit(X_train, y_train)
    
    return final_model, best_params

def evaluate_model(model, X_test, y_test_log, model_name="model"):
    """
    Predict on test set (predictions are in log space).
    Convert predictions back to original price space: np.expm1(preds).
    Convert y_test_log back too: y_true = np.expm1(y_test_log).
    Compute MAE, RMSE, R² on original price scale.
    Print all metrics with model_name prefix.
    Return dict with keys: mae, rmse, r2.
    """
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    y_true = np.expm1(y_test_log)
    
    mae = mean_absolute_error(y_true, preds)
    rmse = mean_squared_error(y_true, preds, squared=False)
    r2 = r2_score(y_true, preds)
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"MAE:  ${mae:.2f}")
    print(f"RMSE: ${rmse:.2f}")
    print(f"R²:   {r2:.4f}")
    
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }

def save_model(model, filepath):
    """
    Save model to filepath using pickle.
    Create parent directory if needed.
    Print confirmation.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to: {filepath}")

def save_metrics(metrics_dict, filepath):
    """
    Save metrics dict as JSON to filepath.
    Create parent directory if needed.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Metrics saved to: {filepath}")

def load_artifacts():
    """
    Load the trained model, the target encoders, and category stats.
    This acts as the single entry point for inference pipelines.
    NOTE: The TargetEncoders MUST be saved alongside the model because they 
    contain the empirical means mapped during training. Failing to use the exact 
    same encoders during inference causes training-serving skew, resulting in garbage predictions.
    """
    with open(config.MODEL_FILE, 'rb') as f:
        model = pickle.load(f)
    encoders = joblib.load(config.MODELS_DIR / "encoders.pkl")
    category_stats = joblib.load(config.MODELS_DIR / "category_stats.pkl")
    return model, encoders, category_stats

def run_training_pipeline():
    """
    Orchestrates data loading, baseline training, LightGBM tuning and training,
    evaluation, and saving model/metrics.
    """
    print("="*60)
    print("DYNAMIC PRICING ENGINE: MODEL TRAINING PIPELINE")
    print("="*60)
    
    # 1. Load data
    train_df, test_df = load_processed_data()
    
    # 2. Split features/target
    X_train, y_train = get_feature_target(train_df)
    X_test, y_test = get_feature_target(test_df)
    
    # 3. Train and Evaluate Baseline
    baseline_model = train_baseline(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_model, X_test, y_test, model_name="Baseline DummyRegressor")
    
    # 4. Train and Evaluate LightGBM
    lgbm_model, best_params = train_lightgbm(X_train, y_train)
    lgbm_metrics = evaluate_model(lgbm_model, X_test, y_test, model_name="LightGBM")
    
    # 5. Compute Lift
    rmse_improvement = ((baseline_metrics['rmse'] - lgbm_metrics['rmse']) / baseline_metrics['rmse']) * 100
    print(f"\nRevenue Lift / Improvement: LightGBM improved RMSE over Baseline by {rmse_improvement:.2f}%")
    
    # 6. Save Artifacts
    save_model(lgbm_model, config.MODEL_FILE)
    
    all_metrics = {
        "baseline": baseline_metrics,
        "lightgbm": lgbm_metrics,
        "improvement_pct": rmse_improvement,
        "best_params": best_params
    }
    save_metrics(all_metrics, config.OUTPUTS_DIR / "metrics.json")
    
    print("\nTraining pipeline finished successfully!")
    return lgbm_model, all_metrics

if __name__ == "__main__":
    run_training_pipeline()
