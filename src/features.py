"""
Feature engineering pipeline for the Dynamic Pricing Engine.

Transforms raw Mercari TSV data into ML-ready features:
  - Category parsing (main / sub / leaf)
  - Text length features (description, name)
  - Brand popularity tier
  - Category-level price statistics
  - Target encoding of categorical columns

Usage:
    python src/features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import TargetEncoder

# ── Resolve project root so the script works when executed directly ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config  # noqa: E402  (must come after sys.path fix)


# ─────────────────────────────────────────────────────────────────────
# 1. Loading
# ─────────────────────────────────────────────────────────────────────

def load_raw_data(filepath: Path) -> pd.DataFrame:
    """Load a raw Mercari TSV file and drop zero-price rows.

    Parameters
    ----------
    filepath : Path
        Absolute or relative path to a ``.tsv`` file.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with all zero-price rows removed.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            "Run the Kaggle download commands from the README first."
        )

    print(f"[load] Reading {filepath.name} …")
    df = pd.read_csv(filepath, sep="\t")
    n_before = len(df)

    # The target column is "price"; drop rows where it is zero or missing
    if "price" in df.columns:
        df = df[df["price"] > 0].copy()
        print(f"[load] Dropped {n_before - len(df):,} rows with price == 0")
    else:
        print("[load] No 'price' column found (test set) — skipping price filter")

    print(f"[load] Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


# ─────────────────────────────────────────────────────────────────────
# 2. Category parsing
# ─────────────────────────────────────────────────────────────────────

def parse_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Split ``category_name`` (e.g. 'Women/Tops/Blouse') into three columns.

    Creates ``category_main``, ``category_sub``, and ``category_leaf``.
    Missing or malformed values are filled with ``'unknown'``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``category_name`` column.

    Returns
    -------
    pd.DataFrame
        Same dataframe with three new category columns added.
    """
    print("[categories] Parsing category_name → main / sub / leaf …")

    # Fill missing before splitting to avoid NaN propagation
    cat = df["category_name"].fillna("unknown/unknown/unknown")

    split = cat.str.split("/", n=2, expand=True)

    df["category_main"] = split[0].fillna("unknown")
    df["category_sub"] = split[1].fillna("unknown") if 1 in split.columns else "unknown"
    df["category_leaf"] = split[2].fillna("unknown") if 2 in split.columns else "unknown"

    n_unknown = (df["category_main"] == "unknown").sum()
    print(f"[categories] Done — {n_unknown:,} rows have unknown main category")
    return df


# ─────────────────────────────────────────────────────────────────────
# 3. Feature engineering
# ─────────────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    price_stats: dict[str, pd.Series] | None = None,
    brand_top_set: set[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.Series], set[str]]:
    """Create all numeric and derived features.

    New columns added:
        - ``log_price``              – log1p(price)
        - ``desc_length``            – character count of item_description
        - ``name_length``            – character count of name
        - ``brand_tier``             – 1 if brand in top 20 % by frequency, else 0
        - ``category_price_median``  – median price within category_main
        - ``category_price_std``     – std of price within category_main

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe that already has parsed categories.
    price_stats : dict or None
        Pre-computed ``{'median': Series, 'std': Series}`` from training
        data.  Pass ``None`` when fitting on train (stats are computed
        fresh).  Pass the returned dict when transforming test data.
    brand_top_set : set or None
        Pre-computed set of top-20 % brand names.  ``None`` → compute
        from this dataframe.

    Returns
    -------
    tuple[pd.DataFrame, dict, set]
        ``(df, price_stats, brand_top_set)`` so callers can reuse the
        fitted statistics on test data.
    """
    print("[features] Engineering features …")

    # ── log target ──────────────────────────────────────────────
    if "price" in df.columns:
        df["log_price"] = np.log1p(df["price"])
        print("[features]  • log_price")

    # ── text length features ───────────────────────────────────
    df["desc_length"] = (
        df["item_description"]
        .fillna("")
        .replace("No description yet", "")
        .str.len()
        .astype(int)
    )
    df["name_length"] = df["name"].fillna("").str.len().astype(int)
    print("[features]  • desc_length, name_length")

    # ── brand tier ─────────────────────────────────────────────
    df["brand_name"] = df["brand_name"].fillna("unknown")

    if brand_top_set is None:
        brand_counts = df["brand_name"].value_counts()
        n_top = max(1, int(len(brand_counts) * 0.20))
        brand_top_set = set(brand_counts.head(n_top).index)

    df["brand_tier"] = df["brand_name"].isin(brand_top_set).astype(int)
    print(f"[features]  • brand_tier  (top-20 % = {len(brand_top_set):,} brands)")

    # ── category price statistics ─────────────────────────────
    if price_stats is None:
        # Fit from training data
        grp = df.groupby("category_main")["price"]
        price_stats = {
            "median": grp.median(),
            "std": grp.std().fillna(0),
        }

    df["category_price_median"] = (
        df["category_main"]
        .map(price_stats["median"])
        .fillna(price_stats["median"].median())  # fallback for unseen categories
    )
    df["category_price_std"] = (
        df["category_main"]
        .map(price_stats["std"])
        .fillna(price_stats["std"].median())
    )
    print("[features]  • category_price_median, category_price_std")

    return df, price_stats, brand_top_set


# ─────────────────────────────────────────────────────────────────────
# 4. Target encoding
# ─────────────────────────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: list[str],
    fit: bool = True,
    encoders: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Target-encode categorical columns using scikit-learn's TargetEncoder.

    When ``fit=True`` the encoders are fitted on ``df`` and the target
    column defined in ``config.TARGET``.  When ``fit=False``, previously
    fitted encoders are applied (for test / inference data).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    cat_cols : list[str]
        Column names to encode.
    fit : bool
        Whether to fit new encoders or reuse existing ones.
    encoders : dict or None
        ``{col_name: fitted TargetEncoder}`` — required when ``fit=False``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        ``(df, encoders)``
    """
    if encoders is None:
        encoders = {}

    target_col = config.TARGET

    for col in cat_cols:
        if col not in df.columns:
            print(f"[encode] WARNING: column '{col}' not found — skipping")
            continue

        # Ensure the column is string type for the encoder
        df[col] = df[col].astype(str)

        if fit:
            if target_col not in df.columns:
                raise ValueError(
                    f"Cannot fit target encoder: target column '{target_col}' "
                    "not found in dataframe."
                )
            enc = TargetEncoder(
                categories="auto",
                smooth="auto",
                target_type="continuous",
                random_state=config.RANDOM_SEED,
            )
            df[col] = enc.fit_transform(
                df[[col]], df[target_col]
            ).ravel()
            encoders[col] = enc
            print(f"[encode] Fitted + transformed: {col}")
        else:
            if col not in encoders:
                raise KeyError(
                    f"No fitted encoder for column '{col}'. "
                    "Pass fit=True first."
                )
            df[col] = encoders[col].transform(df[[col]]).ravel()
            print(f"[encode] Transformed (pre-fitted): {col}")

    return df, encoders


# ─────────────────────────────────────────────────────────────────────
# 5. Full pipeline
# ─────────────────────────────────────────────────────────────────────

def run_pipeline(
    train_path: Path | None = None,
    test_path: Path | None = None,
    save: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Run the complete feature-engineering pipeline end-to-end.

    Steps:
        1. Load raw TSV files
        2. Parse categories
        3. Engineer numeric features (fit on train, apply to test)
        4. Target-encode categoricals (fit on train, apply to test)
        5. Optionally save to parquet

    Parameters
    ----------
    train_path : Path or None
        Path to training TSV.  Defaults to ``config.TRAIN_FILE``.
    test_path : Path or None
        Path to test TSV.  Defaults to ``config.TEST_FILE``.
    save : bool
        If True, write processed dataframes to parquet.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame | None]
        ``(train_df, test_df)`` — ``test_df`` is None when no test file exists.
    """
    train_path = train_path or config.TRAIN_FILE
    test_path = test_path or config.TEST_FILE

    # ── Step 1: Load ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 1 / 5 — Loading raw data")
    print("=" * 60)
    train = load_raw_data(train_path)

    test = None
    if test_path.exists():
        test = load_raw_data(test_path)
    else:
        print(f"[pipeline] Test file not found ({test_path.name}) — skipping")

    # ── Step 2: Parse categories ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 / 5 — Parsing categories")
    print("=" * 60)
    train = parse_categories(train)
    if test is not None:
        test = parse_categories(test)

    # ── Step 3: Engineer features ────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 / 5 — Engineering features")
    print("=" * 60)
    train, price_stats, brand_top_set = engineer_features(train)
    if test is not None:
        test, _, _ = engineer_features(
            test,
            price_stats=price_stats,
            brand_top_set=brand_top_set,
        )

    # ── Step 4: Target encoding ──────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4 / 5 — Target encoding categoricals")
    print("=" * 60)
    train, encoders = encode_categoricals(
        train,
        cat_cols=config.CAT_FEATURES,
        fit=True,
    )
    if test is not None:
        test, _ = encode_categoricals(
            test,
            cat_cols=config.CAT_FEATURES,
            fit=False,
            encoders=encoders,
        )

    # ── Step 5: Save ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 / 5 — Saving processed data")
    print("=" * 60)
    if save:
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

        train.to_parquet(config.PROCESSED_TRAIN, index=False)
        print(f"[save] Train → {config.PROCESSED_TRAIN}")

        if test is not None:
            test.to_parquet(config.PROCESSED_TEST, index=False)
            print(f"[save] Test  → {config.PROCESSED_TEST}")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Train shape : {train.shape}")
    print(f"  Train cols  : {list(train.columns)}")
    if test is not None:
        print(f"  Test shape  : {test.shape}")
        print(f"  Test cols   : {list(test.columns)}")

    return train, test


# ─────────────────────────────────────────────────────────────────────
# Script entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train_df, test_df = run_pipeline()
    print("\n✅ Feature engineering finished successfully.")
