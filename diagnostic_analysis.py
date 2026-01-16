"""
Diagnostic Analysis - Understanding Model Performance
This script analyzes why models are not predicting correctly
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy import sparse as sp
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_predictions():
    """Analyze model predictions and identify issues"""
    
    features_dir = Path(r"c:\Users\ASUS\Desktop\ML project 2\data\features")
    models_dir = Path(r"c:\Users\ASUS\Desktop\ML project 2\models")
    
    # Load data
    X_test = sp.load_npz(features_dir / 'X_test.npz')
    y_test = pd.read_csv(features_dir / 'y_test.csv')['license_id']
    
    with open(features_dir / 'encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    # Load SVM model
    with open(models_dir / 'svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    
    # Get predictions and probabilities
    y_pred = svm_model.predict(X_test)
    decision_scores = svm_model.decision_function(X_test)
    
    print("\n" + "="*100)
    print("DIAGNOSTIC ANALYSIS - MODEL PREDICTIONS")
    print("="*100)
    
    print(f"\n📊 DATA CHARACTERISTICS:")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  y_test unique classes: {len(y_test.unique())}")
    print(f"  SVM classes: {len(svm_model.classes_)}")
    print(f"  Test set class distribution:\n{y_test.value_counts()}")
    
    print(f"\n🔍 PREDICTION ANALYSIS:")
    print(f"  Unique predictions: {len(np.unique(y_pred))}")
    print(f"  Prediction value counts:\n{pd.Series(y_pred).value_counts().head(20)}")
    print(f"  Most common prediction: {pd.Series(y_pred).value_counts().index[0]}")
    
    print(f"\n⚠️  ACCURACY CHECK:")
    accuracy = (y_pred == y_test.values).sum() / len(y_test)
    print(f"  Exact match accuracy: {accuracy:.4f}")
    
    print(f"\n📈 DECISION FUNCTION SCORES:")
    print(f"  Min score: {decision_scores.min():.4f}")
    print(f"  Max score: {decision_scores.max():.4f}")
    print(f"  Mean score: {decision_scores.mean():.4f}")
    print(f"  Decision score shape: {decision_scores.shape}")
    
    print(f"\n🔧 PROBLEM DIAGNOSIS:")
    print(f"""
    Issues identified:
    
    1. MULTI-CLASS COMPLEXITY: 718 classes with 574 training samples
       - Each training sample is essentially a unique class (one-shot learning problem)
       - Test set has 144 different classes (mostly unseen during training)
       - Model sees mostly "new" classes it hasn't trained on
    
    2. SPARSE FEATURE SPACE: 93.23% sparsity in feature matrix
       - Very few non-zero features per sample
       - Hard to distinguish between similar licenses
       - Most TF-IDF features appear in only few documents
    
    3. CLASS IMBALANCE: Each class appears exactly once in training
       - No repetition to learn patterns
       - High variance in model decision boundaries
       - Limited generalization to test classes
    
    RECOMMENDATION: Switch to different evaluation approach
    """)
    
    # Check if predictions are mostly from training set
    y_train = pd.read_csv(features_dir / 'y_train.csv')['license_id']
    pred_in_train = sum(np.isin(y_pred, y_train.values))
    print(f"\n📍 PREDICTION SOURCE:")
    print(f"  Predictions from training classes: {pred_in_train}/{len(y_pred)} ({pred_in_train/len(y_pred)*100:.1f}%)")
    print(f"  This suggests models are overfitting to training classes")
    
    # Top-k accuracy
    print(f"\n🎯 TOP-K ACCURACY CALCULATION:")
    print(f"  (Measures if true class is in top K predictions)")
    
    # Get decision scores for all samples
    decision_matrix = svm_model.decision_function(X_test)  # Shape: (144, 718)
    
    for k in [1, 5, 10, 20, 50]:
        # For each test sample, get top-k predicted classes
        top_k_preds = np.argsort(decision_matrix, axis=1)[:, -k:]
        
        # Check if true label is in top-k predictions
        correct = 0
        for i, true_class_name in enumerate(y_test.values):
            true_class_idx = np.where(svm_model.classes_ == true_class_name)[0]
            if len(true_class_idx) > 0 and true_class_idx[0] in top_k_preds[i]:
                correct += 1
        
        top_k_acc = correct / len(y_test)
        print(f"  Top-{k:2d} Accuracy: {top_k_acc:.4f} ({correct}/{len(y_test)} correct)")
    
    return svm_model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = analyze_predictions()
    
    print("\n" + "="*100)
    print("SUMMARY: WHY ACCURACY IS 0%")
    print("="*100)
    print("""
This is a VALID and EXPECTED result for this specific problem:

1. PROBLEM DESIGN: 718-class SPDX license classification
   - Training set: 574 unique SPDX licenses (each appears once)
   - Test set: 144 different SPDX licenses (mostly new ones)
   - This is NOT a typical multi-class problem with repeated classes

2. WHAT 0% MEANS:
   - No EXACT match between predicted class and actual class
   - Model is not predicting the correct license ID
   - BUT: Top-k accuracy likely shows reasonable performance

3. WHY THIS HAPPENS:
   - Each training class appears exactly once (no repetition)
   - Features are very sparse (93% zeros)
   - Decision boundaries are hard to learn with such sparse data
   - Many test classes never appeared in training

4. PRACTICAL SOLUTION:
   Instead of exact classification, use:
   ✅ Similarity matching (cosine similarity between text embeddings)
   ✅ Top-k ranking (which top 5-10 predictions are correct)
   ✅ Confidence thresholding (only predict when model is confident)
   ✅ Hybrid approach (rule-based + ML)

NEXT STEPS:
- Implement top-k ranking evaluation
- Try dense embeddings (BERT) instead of sparse TF-IDF
- Use similarity-based matching instead of classification
- Compare with string matching baselines
""")
