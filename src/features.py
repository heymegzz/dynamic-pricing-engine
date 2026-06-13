import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import TargetEncoder

sys.path.append(str(Path(__file__).parent.parent))
import config

def load_raw_data(filepath):
    """
    Load CSV with pandas. Drop rows where price == 0. Print shape after loading.
    Return df.
    """
    df = pd.read_csv(filepath, encoding='utf-8', encoding_errors='replace')
    df = df[df['price'] > 0].copy()
    print(f"Loaded data shape after dropping price=0: {df.shape}")
    return df

def parse_categories(df):
    """
    Split category_name column on / into exactly 3 parts: category_main, category_sub, category_leaf.
    If fewer than 3 parts exist, fill missing with "unknown".
    Drop original category_name column.
    Return df.
    """
    # Fill missing categories first to avoid issues during splitting
    df['category_name'] = df['category_name'].fillna('unknown/unknown/unknown')
    
    # Split into 3 columns
    split_cats = df['category_name'].str.split('/', n=2, expand=True)
    
    # Assign and handle missing parts if they didn't have 3 segments
    df['category_main'] = split_cats[0].fillna('unknown')
    df['category_sub'] = split_cats[1].fillna('unknown') if 1 in split_cats.columns else 'unknown'
    df['category_leaf'] = split_cats[2].fillna('unknown') if 2 in split_cats.columns else 'unknown'
    
    df = df.drop(columns=['category_name'])
    return df

def engineer_features(df, category_stats=None):
    """
    Engineer numeric and categorical features.
    Computes log_price, text lengths, brand tier, and category price stats.
    Return df, and category_stats dict (keys: "median", "std", "global_median", "global_std").
    """
    df['log_price'] = np.log1p(df['price'])
    
    # Length of item_description, 0 if null
    df['desc_length'] = df['item_description'].fillna("").str.len()
    
    # Length of name
    df['name_length'] = df['name'].fillna("").str.len()
    
    # Brand tier
    df['brand_name'] = df['brand_name'].fillna('unknown')
    
    # Category price stats & Brand Tier Logic
    if category_stats is None:
        median_series = df.groupby('category_main')['price'].median()
        std_series = df.groupby('category_main')['price'].std().fillna(0)
        global_median = df['price'].median()
        global_std = df['price'].std()
        
        # Calculate top brands on training data only
        brand_counts = df['brand_name'].value_counts()
        top_20_percent_count = max(1, int(len(brand_counts) * 0.2))
        top_brands = set(brand_counts.head(top_20_percent_count).index)
        
        category_stats = {
            "median": median_series.to_dict(),
            "std": std_series.to_dict(),
            "global_median": global_median,
            "global_std": global_std,
            "top_brands": top_brands
        }
    else:
        top_brands = category_stats["top_brands"]
        
    df['brand_tier'] = df['brand_name'].isin(top_brands).astype(int)
    
    df['category_price_median'] = df['category_main'].map(category_stats["median"]).fillna(category_stats["global_median"])
    df['category_price_std'] = df['category_main'].map(category_stats["std"]).fillna(category_stats["global_std"])
    
    return df, category_stats

def encode_categoricals(df, cat_cols, target_col, fit=True, encoders=None):
    """
    Use sklearn TargetEncoder on each cat column individually.
    If fit=True, fit and transform, return (df, encoders dict).
    If fit=False, use provided encoders to transform only.
    Handle unseen categories gracefully.
    Return (df, encoders).
    """
    if encoders is None:
        encoders = {}
        
    for col in cat_cols:
        # Cast categorical column to string to handle mixed types gracefully in TargetEncoder
        df[col] = df[col].astype(str)
        
        if fit:
            enc = TargetEncoder(target_type="continuous", random_state=config.RANDOM_SEED)
            df[col] = enc.fit_transform(df[[col]], df[target_col])
            encoders[col] = enc
        else:
            if col in encoders:
                df[col] = encoders[col].transform(df[[col]])
            else:
                raise ValueError(f"Encoder for {col} not found in provided encoders.")
    
    return df, encoders

def build_inference_row(raw_input: dict, encoders: dict, category_stats: dict) -> np.ndarray:
    """
    Convert a raw user-facing input dict into a (1, n_features) numpy array
    ready for model.predict(). This is the SINGLE source of truth for feature
    construction at inference time — both the API and the dashboard must use
    this function to avoid training/serving skew.

    Args:
        raw_input (dict): Keys: item_name, category_main, category_sub,
            category_leaf, brand_name, item_condition_id, shipping,
            item_description.
        encoders (dict): Fitted TargetEncoder objects, keyed by column name.
        category_stats (dict): Category price stats dict from training.

    Returns:
        np.ndarray: shape (1, len(ALL_FEATURES)), dtype float64.
    """
    # 1. Build a single-row DataFrame mirroring the raw CSV schema
    row = {
        "name":             str(raw_input.get("item_name", "")),
        "category_name":    "/".join([
                                raw_input.get("category_main", "unknown"),
                                raw_input.get("category_sub",  "unknown"),
                                raw_input.get("category_leaf", "unknown"),
                            ]),
        "brand_name":       str(raw_input.get("brand_name", "unknown")),
        "item_condition_id": int(raw_input.get("item_condition_id", 1)),
        "shipping":         int(raw_input.get("shipping", 0)),
        "item_description": str(raw_input.get("item_description", "")),
        "price":            0.0,   # dummy — only needed so engineer_features doesn't fail
    }
    df = pd.DataFrame([row])

    # 2. Parse category_name → category_main, category_sub, category_leaf
    df = parse_categories(df)

    # 3. Engineer numeric features (desc_length, name_length, brand_tier,
    #    category_price_median, category_price_std).  Pass saved category_stats
    #    so we never refit on a single row.
    df, _ = engineer_features(df, category_stats=category_stats)

    # 4. Apply saved TargetEncoders for categorical columns
    df, _ = encode_categoricals(
        df, config.CAT_FEATURES,
        target_col="log_price",
        fit=False,
        encoders=encoders,
    )

    # 5. Assemble columns in the exact order defined by ALL_FEATURES
    feature_row = {k: float(df.iloc[0].get(k, 0.0)) for k in config.ALL_FEATURES}
    return np.array([[feature_row[k] for k in config.ALL_FEATURES]], dtype=np.float64)

import joblib

def run_pipeline(save=True):
    """
    Orchestrate feature engineering pipeline.
    """
    print("Loading raw data...")
    df = load_raw_data(config.TRAIN_FILE)
    
    print("Parsing categories...")
    df = parse_categories(df)
    
    print("Splitting data into train and test (80/20)...")
    train_df, test_df = train_test_split(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)
    
    print("Engineering features for train...")
    train_df, category_stats = engineer_features(train_df, category_stats=None)
    
    print("Engineering features for test...")
    test_df, _ = engineer_features(test_df, category_stats=category_stats)
    
    print("Encoding categoricals for train...")
    train_df, encoders = encode_categoricals(train_df, config.CAT_FEATURES, config.LOG_TARGET, fit=True)
    
    print("Encoding categoricals for test...")
    test_df, _ = encode_categoricals(test_df, config.CAT_FEATURES, config.LOG_TARGET, fit=False, encoders=encoders)
    
    if save:
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(config.PROCESSED_TRAIN, index=False)
        test_df.to_parquet(config.PROCESSED_TEST, index=False)
        print(f"Saved processed data to {config.DATA_PROCESSED}")
        
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, config.MODELS_DIR / "encoders.pkl")
        joblib.dump(category_stats, config.MODELS_DIR / "category_stats.pkl")
        print(f"Saved encoders and category_stats to {config.MODELS_DIR}")
        
    print(f"Final Train shape: {train_df.shape}")
    print(f"Final Test shape: {test_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    print("Pipeline completed successfully!")
    
    return train_df, test_df

if __name__ == "__main__":
    run_pipeline(save=True)
