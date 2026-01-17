"""
Overfitting Detection & Data Leakage Analysis
Check if models are truly generalizing or just memorizing
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")

def load_data():
    """Load all training and test data"""
    logger.info("Loading data...")
    
    X_train = np.load(PROCESSED_DIR / "X_train.npz")['X_train']
    X_test = np.load(PROCESSED_DIR / "X_test.npz")['X_test']
    y_train = np.load(PROCESSED_DIR / "y_train.npz")['y_train']
    y_test = np.load(PROCESSED_DIR / "y_test.npz")['y_test']
    
    df_train = pd.read_csv(PROCESSED_DIR / "train_metadata.csv")
    df_test = pd.read_csv(PROCESSED_DIR / "test_metadata.csv")
    
    logger.info(f"X_train shape: {X_train.shape}")
    logger.info(f"X_test shape: {X_test.shape}")
    logger.info(f"y_train shape: {y_train.shape}")
    logger.info(f"y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test, df_train, df_test

def check_data_leakage(df_train, df_test):
    """Check if training and test data have overlapping license IDs"""
    logger.info("\n" + "="*80)
    logger.info("DATA LEAKAGE CHECK")
    logger.info("="*80)
    
    train_licenses = set(df_train['license_id'].unique())
    test_licenses = set(df_test['license_id'].unique())
    
    overlap = train_licenses & test_licenses
    
    logger.info(f"Training unique licenses: {len(train_licenses)}")
    logger.info(f"Test unique licenses: {len(test_licenses)}")
    logger.info(f"Overlapping licenses: {len(overlap)}")
    
    if overlap:
        logger.warning(f"⚠️ DATA LEAKAGE DETECTED!")
        logger.warning(f"Overlapping license IDs: {list(overlap)[:10]}")
        overlap_samples = df_test[df_test['license_id'].isin(overlap)]
        logger.warning(f"Test samples with overlapping licenses: {len(overlap_samples)}")
        return True
    else:
        logger.info("✓ No direct data leakage (different license_ids in train/test)")
        return False

def check_class_imbalance(y_train, y_test):
    """Check class distribution"""
    logger.info("\n" + "="*80)
    logger.info("CLASS DISTRIBUTION")
    logger.info("="*80)
    
    with open(MODEL_DIR / "label_encoder_license.pkl", "rb") as f:
        le = pickle.load(f)
    
    logger.info("\nTraining set:")
    for cls_idx, cls_name in enumerate(le.classes_):
        count = np.sum(y_train == cls_idx)
        pct = count / len(y_train) * 100
        logger.info(f"  {cls_name:15s}: {count:3d} samples ({pct:5.1f}%)")
    
    logger.info("\nTest set:")
    for cls_idx, cls_name in enumerate(le.classes_):
        count = np.sum(y_test == cls_idx)
        pct = count / len(y_test) * 100
        logger.info(f"  {cls_name:15s}: {count:3d} samples ({pct:5.1f}%)")

def evaluate_model_detailed(model_name, model, X_train, X_test, y_train, y_test):
    """Detailed evaluation of a model"""
    logger.info(f"\n{'='*80}")
    logger.info(f"DETAILED ANALYSIS: {model_name}")
    logger.info(f"{'='*80}")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Per-class accuracy
    logger.info("\nPer-class accuracy (Training):")
    with open(MODEL_DIR / "label_encoder_license.pkl", "rb") as f:
        le = pickle.load(f)
    
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = y_train == cls_idx
        if np.sum(mask) > 0:
            acc = np.mean(y_train_pred[mask] == y_train[mask])
            logger.info(f"  {cls_name:15s}: {acc*100:6.2f}% ({np.sum(mask)} samples)")
    
    logger.info("\nPer-class accuracy (Test):")
    for cls_idx, cls_name in enumerate(le.classes_):
        mask = y_test == cls_idx
        if np.sum(mask) > 0:
            acc = np.mean(y_test_pred[mask] == y_test[mask])
            logger.info(f"  {cls_name:15s}: {acc*100:6.2f}% ({np.sum(mask)} samples)")
    
    # Confusion matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    logger.info("\nConfusion Matrix (Training):")
    logger.info(f"{cm_train}")
    
    logger.info("\nConfusion Matrix (Test):")
    logger.info(f"{cm_test}")
    
    # Check for overfitting
    train_acc = np.mean(y_train_pred == y_train)
    test_acc = np.mean(y_test_pred == y_test)
    gap = (train_acc - test_acc) * 100
    
    logger.info(f"\nAccuracy Gap: {gap:.2f}%")
    if gap > 5:
        logger.warning(f"⚠️ POSSIBLE OVERFITTING (gap > 5%)")
    else:
        logger.info(f"✓ No significant overfitting detected")
    
    return y_train_pred, y_test_pred, cm_train, cm_test

def check_feature_importance(model_name):
    """Check feature importance if available"""
    logger.info(f"\n{'='*80}")
    logger.info(f"FEATURE IMPORTANCE: {model_name}")
    logger.info(f"{'='*80}")
    
    # Convert model_name for file lookup (e.g., "Random Forest" -> "random_forest")
    model_file_name = model_name.lower().replace(' ', '_')
    model_path = MODEL_DIR / f"{model_file_name}_model.pkl"
    
    if not model_path.exists():
        logger.warning(f"Model file not found: {model_path}")
        return
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-10:][::-1]
        
        logger.info(f"\nTop 10 most important features:")
        for rank, idx in enumerate(top_indices, 1):
            logger.info(f"  {rank:2d}. Feature {idx}: {importances[idx]:.6f}")
        
        # Check if importance is concentrated
        total_importance = np.sum(importances)
        top_10_importance = np.sum(importances[top_indices])
        pct = (top_10_importance / total_importance) * 100
        logger.info(f"\nTop 10 features account for {pct:.1f}% of importance")
    else:
        logger.info(f"Model {model_name} does not have feature_importances_")

def main():
    """Main execution"""
    logger.info("\n" + "╔" + "="*78 + "╗")
    logger.info("║" + " "*78 + "║")
    logger.info("║" + "OVERFITTING DETECTION & DATA LEAKAGE ANALYSIS".center(78) + "║")
    logger.info("║" + " "*78 + "║")
    logger.info("╚" + "="*78 + "╝\n")
    
    # Load data
    X_train, X_test, y_train, y_test, df_train, df_test = load_data()
    
    # Check for data leakage
    has_leakage = check_data_leakage(df_train, df_test)
    
    # Check class distribution
    check_class_imbalance(y_train, y_test)
    
    # Evaluate each model
    models = ['random_forest', 'naive_bayes', 'gradient_boosting']
    results = {}
    
    for model_name in models:
        model_path = MODEL_DIR / f"{model_name}_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            
            y_train_pred, y_test_pred, cm_train, cm_test = evaluate_model_detailed(
                model_name.replace('_', ' ').title(), model, X_train, X_test, y_train, y_test
            )
            
            results[model_name] = {
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'cm_train': cm_train.tolist(),
                'cm_test': cm_test.tolist()
            }
            
            # Feature importance
            if model_name == 'random_forest' or model_name == 'gradient_boosting':
                check_feature_importance(model_name.replace('_', ' ').title())
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("OVERFITTING SUMMARY")
    logger.info(f"{'='*80}")
    
    if has_leakage:
        logger.warning("⚠️ CRITICAL: Data leakage detected! Train/test overlap exists.")
        logger.warning("This explains the perfect accuracy.")
    else:
        logger.info("✓ No data leakage (different license IDs)")
        logger.info("\nPossible causes of high accuracy:")
        logger.info("  1. Classes are naturally separable (3 balanced categories)")
        logger.info("  2. TF-IDF features effectively capture category differences")
        logger.info("  3. Models have good generalization on this task")
        logger.info("  4. Dataset is small (717 samples) - easier to achieve high accuracy")
    
    logger.info("\n" + "="*80)
    logger.info("Analysis complete. Check detailed metrics above.")
    logger.info("="*80)

if __name__ == "__main__":
    main()
