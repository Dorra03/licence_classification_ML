"""
Phase 3: Model Training
- Train multiple models (Random Forest, Naive Bayes, XGBoost)
- Save trained models
- Generate training metrics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def load_features():
    """Load pre-computed features"""
    logger.info("Loading features...")
    
    X_train = np.load(PROCESSED_DIR / "X_train.npz")['X_train']
    X_test = np.load(PROCESSED_DIR / "X_test.npz")['X_test']
    y_train = np.load(PROCESSED_DIR / "y_train.npz")['y_train']
    y_test = np.load(PROCESSED_DIR / "y_test.npz")['y_test']
    
    logger.info(f"Loaded training data: {X_train.shape}")
    logger.info(f"Loaded test data: {X_test.shape}")
    logger.info(f"Classes: {len(np.unique(y_train))}")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest"""
    logger.info("\nTraining Random Forest Classifier...")
    
    start_time = time.time()
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'model': 'Random Forest',
        'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
        'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
        'train_precision': float(precision_score(y_train, y_train_pred, average='weighted', zero_division=0)),
        'test_precision': float(precision_score(y_test, y_test_pred, average='weighted', zero_division=0)),
        'train_recall': float(recall_score(y_train, y_train_pred, average='weighted', zero_division=0)),
        'test_recall': float(recall_score(y_test, y_test_pred, average='weighted', zero_division=0)),
        'train_f1': float(f1_score(y_train, y_train_pred, average='weighted', zero_division=0)),
        'test_f1': float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0)),
        'training_time': train_time
    }
    
    logger.info(f"✓ Random Forest Results:")
    logger.info(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    logger.info(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"  Training Time: {train_time:.2f}s")
    
    # Save model
    with open(MODEL_DIR / "random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info(f"  Saved to {MODEL_DIR / 'random_forest_model.pkl'}")
    
    return model, metrics

def train_naive_bayes(X_train, y_train, X_test, y_test):
    """Train Naive Bayes"""
    logger.info("\nTraining Naive Bayes...")
    
    start_time = time.time()
    
    # Convert to non-negative for Multinomial NB
    X_train_nb = np.abs(X_train)
    X_test_nb = np.abs(X_test)
    
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_nb, y_train)
    train_time = time.time() - start_time
    
    # Predictions
    y_train_pred = model.predict(X_train_nb)
    y_test_pred = model.predict(X_test_nb)
    
    # Metrics
    metrics = {
        'model': 'Naive Bayes',
        'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
        'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
        'train_precision': float(precision_score(y_train, y_train_pred, average='weighted', zero_division=0)),
        'test_precision': float(precision_score(y_test, y_test_pred, average='weighted', zero_division=0)),
        'train_recall': float(recall_score(y_train, y_train_pred, average='weighted', zero_division=0)),
        'test_recall': float(recall_score(y_test, y_test_pred, average='weighted', zero_division=0)),
        'train_f1': float(f1_score(y_train, y_train_pred, average='weighted', zero_division=0)),
        'test_f1': float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0)),
        'training_time': train_time
    }
    
    logger.info(f"✓ Naive Bayes Results:")
    logger.info(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    logger.info(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"  Training Time: {train_time:.2f}s")
    
    # Save model
    with open(MODEL_DIR / "naive_bayes_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info(f"  Saved to {MODEL_DIR / 'naive_bayes_model.pkl'}")
    
    return model, metrics

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train Gradient Boosting"""
    logger.info("\nTraining Gradient Boosting...")
    
    start_time = time.time()
    
    model = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Metrics
    metrics = {
        'model': 'Gradient Boosting',
        'train_accuracy': float(accuracy_score(y_train, y_train_pred)),
        'test_accuracy': float(accuracy_score(y_test, y_test_pred)),
        'train_precision': float(precision_score(y_train, y_train_pred, average='weighted', zero_division=0)),
        'test_precision': float(precision_score(y_test, y_test_pred, average='weighted', zero_division=0)),
        'train_recall': float(recall_score(y_train, y_train_pred, average='weighted', zero_division=0)),
        'test_recall': float(recall_score(y_test, y_test_pred, average='weighted', zero_division=0)),
        'train_f1': float(f1_score(y_train, y_train_pred, average='weighted', zero_division=0)),
        'test_f1': float(f1_score(y_test, y_test_pred, average='weighted', zero_division=0)),
        'training_time': train_time
    }
    
    logger.info(f"✓ Gradient Boosting Results:")
    logger.info(f"  Train Accuracy: {metrics['train_accuracy']:.4f}")
    logger.info(f"  Test Accuracy: {metrics['test_accuracy']:.4f}")
    logger.info(f"  Training Time: {train_time:.2f}s")
    
    # Save model
    with open(MODEL_DIR / "gradient_boosting_model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info(f"  Saved to {MODEL_DIR / 'gradient_boosting_model.pkl'}")
    
    return model, metrics

def save_all_metrics(all_metrics):
    """Save all training metrics"""
    logger.info("\nSaving all metrics...")
    
    # Sort by test accuracy
    all_metrics_sorted = sorted(all_metrics, key=lambda x: x['test_accuracy'], reverse=True)
    
    metrics_file = RESULTS_DIR / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics_sorted, f, indent=2)
    
    logger.info(f"✓ Saved to {metrics_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("MODEL COMPARISON (sorted by Test Accuracy)")
    logger.info("=" * 80)
    for metrics in all_metrics_sorted:
        logger.info(f"\n{metrics['model']}:")
        logger.info(f"  Train Acc: {metrics['train_accuracy']:.4f} | Test Acc: {metrics['test_accuracy']:.4f}")
        logger.info(f"  Train F1:  {metrics['train_f1']:.4f} | Test F1:  {metrics['test_f1']:.4f}")
        logger.info(f"  Time: {metrics['training_time']:.2f}s")
    
    return all_metrics_sorted

def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("PHASE 3: MODEL TRAINING")
    logger.info("=" * 80)
    
    try:
        # Load features
        X_train, X_test, y_train, y_test = load_features()
        
        all_metrics = []
        
        # Train models
        rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
        all_metrics.append(rf_metrics)
        
        nb_model, nb_metrics = train_naive_bayes(X_train, y_train, X_test, y_test)
        all_metrics.append(nb_metrics)
        
        gb_model, gb_metrics = train_gradient_boosting(X_train, y_train, X_test, y_test)
        all_metrics.append(gb_metrics)
        
        # Save all metrics
        all_metrics_sorted = save_all_metrics(all_metrics)
        
        best_model = all_metrics_sorted[0]['model']
        best_accuracy = all_metrics_sorted[0]['test_accuracy']
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ PHASE 3: MODEL TRAINING COMPLETE")
        logger.info(f"  Best Model: {best_model} ({best_accuracy:.4f})")
        logger.info(f"  Models Saved: {MODEL_DIR}")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in model training: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
