"""
Phase 3, Step 5+6: Feature Engineering and Data Preparation
- Extract TF-IDF features (matching Approach 1)
- Add categorical features
- Create train/test splits
- Prepare for model training
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def load_normalized_data():
    """Load normalized dataset"""
    logger.info("Loading normalized data...")
    
    csv_file = PROCESSED_DIR / "normalized_dataset.csv"
    if not csv_file.exists():
        logger.error(f"Normalized data not found at {csv_file}")
        logger.error("Please run Step 4 first!")
        return None
    
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} licenses")
    
    return df

def extract_tfidf_features(df, n_features=5000):
    """Extract TF-IDF features from license texts"""
    logger.info(f"Extracting TF-IDF features ({n_features} dimensions)...")
    
    texts = df['text'].fillna('')
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=n_features,
        lowercase=True,
        stop_words='english',
        min_df=1,
        max_df=0.95,
        ngram_range=(1, 2)
    )
    
    # Fit and transform
    X_tfidf = vectorizer.fit_transform(texts)
    
    logger.info(f"✓ TF-IDF features shape: {X_tfidf.shape}")
    logger.info(f"  Sparsity: {1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.2%}")
    
    # Save vectorizer
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Saved vectorizer to {MODEL_DIR / 'tfidf_vectorizer.pkl'}")
    
    return X_tfidf, vectorizer

def add_categorical_features(df, X_tfidf):
    """
    REMOVED: Categorical features should NOT be used as input!
    The category field is the TARGET variable, not a feature.
    Using it as a feature = data leakage (model learns to memorize the target).
    """
    logger.info("⚠️  Skipping categorical features (they are TARGET variables, not inputs)")
    logger.info("    Using ONLY TF-IDF features for training")
    return None, None

def combine_features(X_tfidf, X_categorical):
    """Combine TF-IDF features ONLY (categorical features cause data leakage)"""
    logger.info("Preparing final feature matrix...")
    
    # Convert to dense for better memory efficiency during training
    if hasattr(X_tfidf, 'toarray'):
        X = X_tfidf.toarray()
    else:
        X = X_tfidf
    
    logger.info(f"✓ Final feature matrix shape: {X.shape}")
    logger.info(f"  Dimensions: {X.shape[1]} (TF-IDF only)")
    logger.info(f"  ℹ️  Categorical features EXCLUDED (they are target labels)")
    
    return X

def prepare_labels(df):
    """Prepare target labels - using category for balanced classification"""
    logger.info("Preparing labels...")
    
    # Use category (permissive, copyleft, other) as the target
    # This provides balanced classes with proper generalization
    le_license = LabelEncoder()
    y = le_license.fit_transform(df['category'].fillna('other'))
    
    unique_classes = len(np.unique(y))
    logger.info(f"✓ {unique_classes} license categories")
    
    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    for cls, count in zip(le_license.classes_, counts):
        logger.info(f"   - {cls}: {count} samples ({count/len(y)*100:.1f}%)")
    
    # Save label encoder
    with open(MODEL_DIR / "label_encoder_license.pkl", "wb") as f:
        pickle.dump(le_license, f)
    
    return y, le_license

def create_train_test_split(X, y, df, test_size=0.2, random_state=42):
    """Create stratified train/test split"""
    logger.info(f"Creating train/test split ({1-test_size:.0%}/{test_size:.0%})...")
    
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=y  # Stratified by category labels (now the target)
    )
    
    logger.info(f"✓ Training set: {X_train.shape[0]} samples")
    logger.info(f"✓ Test set: {X_test.shape[0]} samples")
    
    # Get dataframe subsets
    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    
    logger.info(f"  Train categories: {dict(df_train['category'].value_counts())}")
    logger.info(f"  Test categories: {dict(df_test['category'].value_counts())}")
    
    return X_train, X_test, y_train, y_test, df_train, df_test

def save_features(X_train, X_test, y_train, y_test, df_train, df_test):
    """Save feature matrices and labels"""
    logger.info("Saving features...")
    
    # Save as NPZ (numpy binary format)
    np.savez_compressed(
        PROCESSED_DIR / "X_train.npz",
        X_train=X_train.astype(np.float32)
    )
    np.savez_compressed(
        PROCESSED_DIR / "X_test.npz",
        X_test=X_test.astype(np.float32)
    )
    np.savez_compressed(
        PROCESSED_DIR / "y_train.npz",
        y_train=y_train
    )
    np.savez_compressed(
        PROCESSED_DIR / "y_test.npz",
        y_test=y_test
    )
    
    logger.info(f"✓ Features saved to {PROCESSED_DIR}")
    
    # Save dataframes
    df_train.to_csv(PROCESSED_DIR / "train_metadata.csv", index=False)
    df_test.to_csv(PROCESSED_DIR / "test_metadata.csv", index=False)
    
    logger.info(f"✓ Metadata saved")

def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("PHASE 3: FEATURE ENGINEERING")
    logger.info("=" * 80)
    
    try:
        # Load data
        df = load_normalized_data()
        if df is None:
            logger.error("Failed to load data")
            return False
        
        # Extract features
        X_tfidf, vectorizer = extract_tfidf_features(df, n_features=5000)
        X_categorical, le_category = add_categorical_features(df, X_tfidf)
        X = combine_features(X_tfidf, X_categorical)
        
        # Prepare labels
        y, le_license = prepare_labels(df)
        
        # Create split
        X_train, X_test, y_train, y_test, df_train, df_test = create_train_test_split(X, y, df)
        
        # Save
        save_features(X_train, X_test, y_train, y_test, df_train, df_test)
        
        # Save feature info
        feature_info = {
            'total_samples': len(df),
            'train_samples': len(df_train),
            'test_samples': len(df_test),
            'feature_dimensions': X.shape[1],
            'n_tfidf_features': 5000,
            'n_categorical_features': 0,
            'n_classes': len(np.unique(y)),
            'data_leakage_fixed': True,
            'note': 'Categorical features removed - they were the target variable!'
        }
        
        with open(PROCESSED_DIR / "feature_info.json", 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("✓ PHASE 3: FEATURE ENGINEERING COMPLETE")
        logger.info(f"  Total features: {X.shape[1]} (TF-IDF only)")
        logger.info(f"  Training samples: {len(df_train)}")
        logger.info(f"  Test samples: {len(df_test)}")
        logger.info(f"  License classes: {len(np.unique(y))}")
        logger.info(f"  ✓ Data leakage fixed (removed categorical feature)")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

