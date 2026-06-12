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
    brand_counts = df['brand_name'].value_counts()
    top_20_percent_count = max(1, int(len(brand_counts) * 0.2))
    top_brands = set(brand_counts.head(top_20_percent_count).index)
    df['brand_tier'] = df['brand_name'].isin(top_brands).astype(int)
    
    # Category price stats
    if category_stats is None:
        median_series = df.groupby('category_main')['price'].median()
        std_series = df.groupby('category_main')['price'].std().fillna(0)
        global_median = df['price'].median()
        global_std = df['price'].std()
        
        category_stats = {
            "median": median_series.to_dict(),
            "std": std_series.to_dict(),
            "global_median": global_median,
            "global_std": global_std
        }
    
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
    train_df, encoders = encode_categoricals(train_df, config.CAT_FEATURES, config.TARGET, fit=True)
    
    print("Encoding categoricals for test...")
    test_df, _ = encode_categoricals(test_df, config.CAT_FEATURES, config.TARGET, fit=False, encoders=encoders)
    
    if save:
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(config.PROCESSED_TRAIN, index=False)
        test_df.to_parquet(config.PROCESSED_TEST, index=False)
        print(f"Saved processed data to {config.DATA_PROCESSED}")
        
    print(f"Final Train shape: {train_df.shape}")
    print(f"Final Test shape: {test_df.shape}")
    print(f"Columns: {list(train_df.columns)}")
    print("Pipeline completed successfully!")
    
    return train_df, test_df

if __name__ == "__main__":
    run_pipeline(save=True)
