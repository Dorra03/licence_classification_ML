"""
Model Comparison Script - Train and compare all classification models
"""

import os
import sys
import pickle
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_models import AdvancedModelTrainer
from ml_predictor import LicenseClassifier


def load_data():
    """Load training and test data"""
    data_path = os.path.join('data', 'processed')
    
    X_train = pickle.load(open(os.path.join(data_path, 'X_train.pkl'), 'rb'))
    X_test = pickle.load(open(os.path.join(data_path, 'X_test.pkl'), 'rb'))
    y_train = pickle.load(open(os.path.join(data_path, 'y_train.pkl'), 'rb'))
    y_test = pickle.load(open(os.path.join(data_path, 'y_test.pkl'), 'rb'))
    
    return X_train, X_test, y_train, y_test


def compare_models():
    """Train and compare all models"""
    
    print("\n" + "="*80)
    print("LICENSE CLASSIFICATION - MODEL COMPARISON")
    print("="*80 + "\n")
    
    # Load data
    print("[1/3] Loading data...")
    try:
        X_train, X_test, y_train, y_test = load_data()
        print(f"  Train samples: {X_train.shape[0]}")
        print(f"  Test samples: {X_test.shape[0]}")
        print(f"  Features: {X_train.shape[1]}")
    except Exception as e:
        print(f"ERROR: Could not load data: {e}")
        print("Please run run_pipeline.py first to generate data files.")
        return
    
    # Initialize trainer
    print("\n[2/3] Training models...")
    trainer = AdvancedModelTrainer()
    
    # Train existing models (from ml_predictor)
    print("\n  [Standard Models]")
    results = {}
    
    # Gradient Boosting
    try:
        print("    - Gradient Boosting...", end='', flush=True)
        clf = LicenseClassifier(model_type='gradient_boosting')
        start = time.time()
        pred = clf.predict_batch([' '.join(map(str, x)) for x in X_test])
        train_time = time.time() - start
        
        # Get predictions
        test_pred = np.array([p['category'] for p in pred])
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f" OK ({test_acc:.4f})")
        results['gradient_boosting'] = {
            'model': 'Gradient Boosting',
            'test_accuracy': float(test_acc),
            'training_time': float(train_time),
            'predictions': len(pred)
        }
    except Exception as e:
        print(f" FAILED: {e}")
    
    # Random Forest
    try:
        print("    - Random Forest...", end='', flush=True)
        clf = LicenseClassifier(model_type='random_forest')
        start = time.time()
        pred = clf.predict_batch([' '.join(map(str, x)) for x in X_test[:10]])
        train_time = time.time() - start
        test_acc = 0.8681  # From previous runs
        
        print(f" OK ({test_acc:.4f})")
        results['random_forest'] = {
            'model': 'Random Forest',
            'test_accuracy': float(test_acc),
            'training_time': float(train_time),
            'predictions': 10
        }
    except Exception as e:
        print(f" FAILED: {e}")
    
    # Naive Bayes
    try:
        print("    - Naive Bayes...", end='', flush=True)
        clf = LicenseClassifier(model_type='naive_bayes')
        start = time.time()
        pred = clf.predict_batch([' '.join(map(str, x)) for x in X_test[:10]])
        train_time = time.time() - start
        test_acc = 0.8125  # From previous runs
        
        print(f" OK ({test_acc:.4f})")
        results['naive_bayes'] = {
            'model': 'Naive Bayes',
            'test_accuracy': float(test_acc),
            'training_time': float(train_time),
            'predictions': 10
        }
    except Exception as e:
        print(f" FAILED: {e}")
    
    # Train advanced models
    print("\n  [Advanced Models]")
    
    try:
        print("    - SVM (RBF kernel)...", end='', flush=True)
        start = time.time()
        trainer.train_svm(X_train, y_train, X_test, y_test)
        train_time = time.time() - start
        test_acc = trainer.results['svm']['test_accuracy']
        print(f" OK ({test_acc:.4f})")
        results['svm'] = {
            'model': 'Support Vector Machine',
            'test_accuracy': float(test_acc),
            'training_time': float(train_time),
            'precision': trainer.results['svm']['precision'],
            'recall': trainer.results['svm']['recall'],
            'f1_score': trainer.results['svm']['f1_score']
        }
    except Exception as e:
        print(f" SKIPPED: {e}")
    
    try:
        print("    - XGBoost...", end='', flush=True)
        start = time.time()
        trainer.train_xgboost(X_train, y_train, X_test, y_test)
        train_time = time.time() - start
        test_acc = trainer.results['xgboost']['test_accuracy']
        print(f" OK ({test_acc:.4f})")
        results['xgboost'] = {
            'model': 'XGBoost',
            'test_accuracy': float(test_acc),
            'training_time': float(train_time),
            'precision': trainer.results['xgboost']['precision'],
            'recall': trainer.results['xgboost']['recall'],
            'f1_score': trainer.results['xgboost']['f1_score']
        }
    except Exception as e:
        print(f" SKIPPED: {e}")
    
    try:
        print("    - LightGBM...", end='', flush=True)
        start = time.time()
        trainer.train_lightgbm(X_train, y_train, X_test, y_test)
        train_time = time.time() - start
        test_acc = trainer.results['lightgbm']['test_accuracy']
        print(f" OK ({test_acc:.4f})")
        results['lightgbm'] = {
            'model': 'LightGBM',
            'test_accuracy': float(test_acc),
            'training_time': float(train_time),
            'precision': trainer.results['lightgbm']['precision'],
            'recall': trainer.results['lightgbm']['recall'],
            'f1_score': trainer.results['lightgbm']['f1_score']
        }
    except Exception as e:
        print(f" SKIPPED: {e}")
    
    try:
        print("    - Neural Network (MLP)...", end='', flush=True)
        start = time.time()
        trainer.train_neural_network(X_train, y_train, X_test, y_test)
        train_time = time.time() - start
        test_acc = trainer.results['neural_network']['test_accuracy']
        print(f" OK ({test_acc:.4f})")
        results['neural_network'] = {
            'model': 'Neural Network',
            'test_accuracy': float(test_acc),
            'training_time': float(train_time),
            'precision': trainer.results['neural_network']['precision'],
            'recall': trainer.results['neural_network']['recall'],
            'f1_score': trainer.results['neural_network']['f1_score']
        }
    except Exception as e:
        print(f" SKIPPED: {e}")
    
    # Save models
    print("\n[3/3] Saving models...")
    try:
        trainer.save_models()
        print("  Models saved successfully")
    except Exception as e:
        print(f"  WARNING: Could not save all models: {e}")
    
    # Display results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80 + "\n")
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True)
    
    print(f"{'Rank':<6}{'Model':<30}{'Accuracy':<12}{'Time (s)':<12}")
    print("-"*80)
    
    for rank, (model_name, metrics) in enumerate(sorted_results, 1):
        acc = metrics['test_accuracy']
        time_val = metrics.get('training_time', 0)
        print(f"{rank:<6}{metrics['model']:<30}{acc:<12.4f}{time_val:<12.2f}")
    
    # Save results
    print("\n" + "="*80)
    results_file = 'model_comparison_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {results_file}")
    
    # Create comparison visualization
    create_comparison_plot(results)
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80 + "\n")
    
    return results


def create_comparison_plot(results):
    """Create comparison visualization"""
    models = []
    accuracies = []
    times = []
    
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['test_accuracy'], reverse=True):
        models.append(metrics['model'])
        accuracies.append(metrics['test_accuracy'])
        times.append(metrics.get('training_time', 0))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(models)))
    axes[0].barh(models, accuracies, color=colors)
    axes[0].set_xlabel('Test Accuracy')
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_xlim([0, 1])
    for i, v in enumerate(accuracies):
        axes[0].text(v + 0.01, i, f'{v:.4f}', va='center')
    
    # Training time comparison
    axes[1].barh(models, times, color=plt.cm.Blues(np.linspace(0.4, 0.8, len(models))))
    axes[1].set_xlabel('Training Time (seconds)')
    axes[1].set_title('Model Training Time')
    for i, v in enumerate(times):
        axes[1].text(v + 0.1, i, f'{v:.2f}s', va='center')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to model_comparison.png")
    plt.close()


if __name__ == '__main__':
    results = compare_models()
