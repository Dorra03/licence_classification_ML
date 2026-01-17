"""
Approach 2: Comprehensive Visualization Suite
Creates detailed visualizations of model performance, confusion matrices, ROC curves, etc.
"""

import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Setup
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data():
    """Load training and test data"""
    X_train = np.load(PROCESSED_DIR / "X_train.npz")['X_train']
    X_test = np.load(PROCESSED_DIR / "X_test.npz")['X_test']
    y_train = np.load(PROCESSED_DIR / "y_train.npz")['y_train']
    y_test = np.load(PROCESSED_DIR / "y_test.npz")['y_test']
    
    with open(MODEL_DIR / "label_encoder_license.pkl", "rb") as f:
        le = pickle.load(f)
    
    return X_train, X_test, y_train, y_test, le

def load_models():
    """Load trained models"""
    with open(MODEL_DIR / "random_forest_model.pkl", "rb") as f:
        rf = pickle.load(f)
    with open(MODEL_DIR / "naive_bayes_model.pkl", "rb") as f:
        nb = pickle.load(f)
    with open(MODEL_DIR / "gradient_boosting_model.pkl", "rb") as f:
        gb = pickle.load(f)
    
    return {'Random Forest': rf, 'Naive Bayes': nb, 'Gradient Boosting': gb}

# 1. CONFUSION MATRICES
def plot_confusion_matrices(X_train, X_test, y_train, y_test, models, le):
    """Create confusion matrices for all models"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold', y=1.00)
    
    for idx, (name, model) in enumerate(models.items()):
        # Training
        y_train_pred = model.predict(X_train)
        cm_train = confusion_matrix(y_train, y_train_pred)
        
        ax = axes[0, idx]
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                   xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_title(f'{name} - Training', fontweight='bold')
        ax.set_ylabel('True Label')
        if idx == 0:
            ax.set_ylabel('True Label')
        
        # Testing
        y_test_pred = model.predict(X_test)
        cm_test = confusion_matrix(y_test, y_test_pred)
        
        ax = axes[1, idx]
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=ax, cbar=False,
                   xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_title(f'{name} - Testing', fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: confusion_matrices.png")
    plt.close()

# 2. ROC CURVES
def plot_roc_curves(X_test, y_test, models, le):
    """Create ROC curves for all models (one-vs-rest)"""
    n_classes = len(le.classes_)
    y_test_bin = label_binarize(y_test, classes=np.arange(n_classes))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('ROC Curves (One-vs-Rest) - Test Set', fontsize=14, fontweight='bold')
    
    colors = sns.color_palette("husl", n_classes)
    
    for idx, (name, model) in enumerate(models.items()):
        ax = axes[idx]
        
        # Get probabilities
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        else:
            # For some models that don't have predict_proba
            y_pred = model.predict(X_test)
            y_proba = np.eye(n_classes)[y_pred]
        
        # Compute ROC curve and AUC for each class
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{le.classes_[i]} (AUC={roc_auc:.3f})', 
                   color=colors[i], linewidth=2)
        
        # Diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{name}', fontweight='bold')
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'roc_curves.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: roc_curves.png")
    plt.close()

# 3. PER-CLASS ACCURACY
def plot_per_class_accuracy(X_train, X_test, y_train, y_test, models, le):
    """Compare per-class accuracy across models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Per-Class Accuracy Comparison', fontsize=14, fontweight='bold')
    
    results_train = {name: [] for name in models.keys()}
    results_test = {name: [] for name in models.keys()}
    
    for name, model in models.items():
        # Training
        y_train_pred = model.predict(X_train)
        for i in range(len(le.classes_)):
            mask = y_train == i
            acc = np.mean(y_train_pred[mask] == y_train[mask])
            results_train[name].append(acc)
        
        # Testing
        y_test_pred = model.predict(X_test)
        for i in range(len(le.classes_)):
            mask = y_test == i
            acc = np.mean(y_test_pred[mask] == y_test[mask])
            results_test[name].append(acc)
    
    x = np.arange(len(le.classes_))
    width = 0.25
    
    # Training
    ax = axes[0]
    for i, name in enumerate(models.keys()):
        ax.bar(x + i*width, results_train[name], width, label=name, alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Training Set', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(le.classes_)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Testing
    ax = axes[1]
    for i, name in enumerate(models.keys()):
        ax.bar(x + i*width, results_test[name], width, label=name, alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Test Set', fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(le.classes_)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: per_class_accuracy.png")
    plt.close()

# 4. TRAIN VS TEST ACCURACY
def plot_train_test_comparison(X_train, X_test, y_train, y_test, models, le):
    """Compare training vs testing accuracy"""
    train_accs = []
    test_accs = []
    model_names = []
    
    for name, model in models.items():
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_acc = np.mean(y_train_pred == y_train)
        test_acc = np.mean(y_test_pred == y_test)
        
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        model_names.append(name)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(model_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_accs, width, label='Training', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Testing', alpha=0.8, color='lightcoral')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add overfitting gap
    for i, (train, test) in enumerate(zip(train_accs, test_accs)):
        gap = train - test
        ax.text(i, max(train, test) + 0.02, f'Gap: {gap:.1%}', 
               ha='center', fontsize=9, color='red', fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Train vs Test Accuracy (Overfitting Analysis)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(fontsize=11)
    ax.set_ylim([0.7, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'train_test_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: train_test_comparison.png")
    plt.close()

# 5. FEATURE IMPORTANCE
def plot_feature_importance(models):
    """Plot feature importance for tree-based models"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Top 15 Most Important Features', fontsize=14, fontweight='bold')
    
    # Random Forest
    rf = models['Random Forest']
    if hasattr(rf, 'feature_importances_'):
        importances = rf.feature_importances_
        top_indices = np.argsort(importances)[-15:][::-1]
        
        ax = axes[0]
        ax.barh(range(len(top_indices)), importances[top_indices], alpha=0.8, color='steelblue')
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels([f'Feature {i}' for i in top_indices])
        ax.set_xlabel('Importance')
        ax.set_title('Random Forest', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    
    # Gradient Boosting
    gb = models['Gradient Boosting']
    if hasattr(gb, 'feature_importances_'):
        importances = gb.feature_importances_
        top_indices = np.argsort(importances)[-15:][::-1]
        
        ax = axes[1]
        ax.barh(range(len(top_indices)), importances[top_indices], alpha=0.8, color='seagreen')
        ax.set_yticks(range(len(top_indices)))
        ax.set_yticklabels([f'Feature {i}' for i in top_indices])
        ax.set_xlabel('Importance')
        ax.set_title('Gradient Boosting', fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_importance.png")
    plt.close()

# 6. PERFORMANCE METRICS HEATMAP
def plot_metrics_heatmap(X_train, X_test, y_train, y_test, models, le):
    """Create heatmap of various metrics"""
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    metrics = {}
    
    for name, model in models.items():
        y_test_pred = model.predict(X_test)
        
        precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
        
        train_acc = np.mean(model.predict(X_train) == y_train)
        test_acc = np.mean(y_test_pred == y_test)
        
        metrics[name] = {
            'Train Acc': train_acc,
            'Test Acc': test_acc,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        }
    
    df = pd.DataFrame(metrics).T
    
    fig, ax = plt.subplots(figsize=(10, 4))
    
    sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.7, vmax=1.0,
               cbar_kws={'label': 'Score'}, ax=ax, linewidths=0.5)
    
    ax.set_title('Performance Metrics Summary (Test Set)', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('Model')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'metrics_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: metrics_heatmap.png")
    plt.close()

# 7. CLASS DISTRIBUTION
def plot_class_distribution(y_train, y_test, le):
    """Plot class distribution in train/test"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Class Distribution', fontsize=14, fontweight='bold')
    
    # Training
    unique, counts = np.unique(y_train, return_counts=True)
    colors = sns.color_palette("husl", len(le.classes_))
    
    ax = axes[0]
    wedges, texts, autotexts = ax.pie(counts, labels=le.classes_, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    ax.set_title(f'Training Set (n={len(y_train)})', fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Testing
    unique, counts = np.unique(y_test, return_counts=True)
    
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(counts, labels=le.classes_, autopct='%1.1f%%',
                                        colors=colors, startangle=90)
    ax.set_title(f'Test Set (n={len(y_test)})', fontweight='bold')
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'class_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: class_distribution.png")
    plt.close()

def main():
    """Generate all visualizations"""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS FOR APPROACH 2")
    print("="*80 + "\n")
    
    # Load data
    print("Loading data and models...")
    X_train, X_test, y_train, y_test, le = load_data()
    models = load_models()
    
    print(f"✓ Loaded {len(X_train)} training and {len(X_test)} test samples")
    print(f"✓ Loaded {len(models)} models\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_class_distribution(y_train, y_test, le)
    plot_confusion_matrices(X_train, X_test, y_train, y_test, models, le)
    plot_per_class_accuracy(X_train, X_test, y_train, y_test, models, le)
    plot_train_test_comparison(X_train, X_test, y_train, y_test, models, le)
    plot_roc_curves(X_test, y_test, models, le)
    plot_feature_importance(models)
    plot_metrics_heatmap(X_train, X_test, y_train, y_test, models, le)
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS GENERATED")
    print("="*80)
    print(f"\nSaved to: {RESULTS_DIR}/")
    print("\nGenerated files:")
    for f in sorted(RESULTS_DIR.glob("*.png")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
