"""
Diagnose data issues: check preprocessing, features, and train/test split
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import load_npz

print("=" * 80)
print("DATA DIAGNOSIS: Checking for fundamental issues")
print("=" * 80)

# ============================================================================
# 1. CHECK RAW CLEANED DATA
# ============================================================================
print("\n[1] RAW CLEANED DATA")
print("-" * 80)

cleaned_file = Path('data/processed/licenses_cleaned.csv')
if cleaned_file.exists():
    df = pd.read_csv(cleaned_file)
    print(f"✓ Loaded: {cleaned_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Unique license_ids: {df['license_id'].nunique()}")
    print(f"\n  Sample licenses:")
    print(f"    {list(df['license_id'].unique()[:5])}")
    
    # Check for duplicates
    dup_count = df.duplicated(subset=['license_id']).sum()
    print(f"\n  Duplicates (license_id): {dup_count}")
    
    # Check text quality
    print(f"\n  Text quality:")
    print(f"    Min text length: {df['cleaned_text'].str.len().min()} chars")
    print(f"    Max text length: {df['cleaned_text'].str.len().max()} chars")
    print(f"    Avg text length: {df['cleaned_text'].str.len().mean():.0f} chars")
    
    # Check for empty/null text
    empty_text = (df['cleaned_text'].str.len() == 0).sum()
    null_text = df['cleaned_text'].isnull().sum()
    print(f"    Empty text: {empty_text}, Null text: {null_text}")
    
    print(f"\n  Sample text (first 300 chars):")
    print(f"    {df['cleaned_text'].iloc[0][:300]}")
else:
    print(f"✗ File not found: {cleaned_file}")
    df = None

# ============================================================================
# 2. CHECK TRAIN DATA
# ============================================================================
print("\n[2] TRAINING DATA")
print("-" * 80)

try:
    y_train = pd.read_csv('data/features/y_train.csv', header=None)
    if len(y_train.columns) == 1:
        y_train.columns = ['license_id']
    X_train = None  # No NPZ files, will note this
    
    print(f"✓ Loaded training data")
    print(f"  Samples: {len(y_train)}")
    if X_train is not None:
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Sparsity: {100 * (1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1])):.2f}%")
    print(f"  Unique classes: {y_train['license_id'].nunique()}")
    print(f"\n  First 10 classes: {list(y_train['license_id'].values[:10])}")
    print(f"\n  Class value counts (first 5):")
    print(y_train['license_id'].value_counts().head(5))
except Exception as e:
    print(f"✗ Error loading training data: {e}")
    y_train = None
    X_train = None

# ============================================================================
# 3. CHECK TEST DATA
# ============================================================================
print("\n[3] TEST DATA")
print("-" * 80)

try:
    y_test = pd.read_csv('data/features/y_test.csv', header=None)
    if len(y_test.columns) == 1:
        y_test.columns = ['license_id']
    X_test = None  # No NPZ files
    
    print(f"✓ Loaded test data")
    print(f"  Samples: {len(y_test)}")
    if X_test is not None:
        print(f"  Features: {X_test.shape[1]}")
        print(f"  Sparsity: {100 * (1 - X_test.nnz / (X_test.shape[0] * X_test.shape[1])):.2f}%")
    print(f"  Unique classes: {y_test['license_id'].nunique()}")
    print(f"\n  First 10 classes: {list(y_test['license_id'].values[:10])}")
    print(f"\n  Class value counts (first 5):")
    print(y_test['license_id'].value_counts().head(5))
except Exception as e:
    print(f"✗ Error loading test data: {e}")
    y_test = None
    X_test = None

# ============================================================================
# 4. CHECK CLASS OVERLAP & SPLIT VALIDITY
# ============================================================================
print("\n[4] CLASS OVERLAP & SPLIT VALIDITY")
print("-" * 80)

if y_train is not None and y_test is not None:
    train_classes = set(y_train['license_id'].values)
    test_classes = set(y_test['license_id'].values)
    all_classes = set(df['license_id'].unique()) if df is not None else set()
    
    overlap = train_classes & test_classes
    only_train = train_classes - test_classes
    only_test = test_classes - train_classes
    
    print(f"Total unique classes (raw data): {len(all_classes)}")
    print(f"Train classes: {len(train_classes)}")
    print(f"Test classes: {len(test_classes)}")
    print(f"\nOverlap: {len(overlap)} classes")
    print(f"Only in train: {len(only_train)} classes")
    print(f"Only in test: {len(only_test)} classes")
    
    if len(overlap) == 0:
        print("\n⚠️  WARNING: NO CLASS OVERLAP - Test set has completely different classes!")
        print("   This is the core issue!")
    
    # Check if train samples are repeated
    train_duplicates = y_train['license_id'].duplicated().sum()
    test_duplicates = y_test['license_id'].duplicated().sum()
    
    print(f"\nDuplicate classes in train: {train_duplicates}")
    print(f"Duplicate classes in test: {test_duplicates}")
    
    if train_duplicates == 0 and test_duplicates == 0:
        print("⚠️  Each class appears exactly once - one-shot learning problem!")

# ============================================================================
# 5. COMPARE RAW VS PROCESSED DATA
# ============================================================================
print("\n[5] DATA INTEGRITY CHECK")
print("-" * 80)

if df is not None and y_train is not None and y_test is not None:
    raw_classes = set(df['license_id'].unique())
    processed_classes = train_classes | test_classes
    
    missing_in_processed = raw_classes - processed_classes
    extra_in_processed = processed_classes - raw_classes
    
    print(f"Raw data classes: {len(raw_classes)}")
    print(f"Processed data classes: {len(processed_classes)}")
    print(f"Match: {raw_classes == processed_classes}")
    
    if len(missing_in_processed) > 0:
        print(f"\n✗ Missing in processed (dropped): {len(missing_in_processed)}")
        print(f"  Examples: {list(missing_in_processed)[:5]}")
    
    if len(extra_in_processed) > 0:
        print(f"\n✗ Extra in processed: {len(extra_in_processed)}")
        print(f"  Examples: {list(extra_in_processed)[:5]}")
    
    # Check sample counts
    raw_samples = len(df)
    processed_samples = len(y_train) + len(y_test)
    print(f"\nRaw data samples: {raw_samples}")
    print(f"Processed data samples: {processed_samples}")
    print(f"Match: {raw_samples == processed_samples}")

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================
print("\n[6] ANALYSIS & RECOMMENDATIONS")
print("=" * 80)

if y_train is not None and y_test is not None and len(overlap) == 0:
    print("""
✗ FUNDAMENTAL PROBLEM IDENTIFIED:
  
  The train/test split was done RANDOMLY without ensuring class overlap.
  Result: Test set has 144 completely NEW classes that don't exist in training.
  
  This is why you get 0% accuracy - the model has never seen these classes!

SOLUTIONS:

  Option 1: STRATIFIED SPLIT (WRONG for this problem)
  ❌ Why: With 718 classes and only 718 samples, stratified split creates
     separate classes in train/test (current problem)

  Option 2: PROPER TRAIN/TEST SPLIT (RECOMMENDED)
  ✓ Sample multiple examples per class, then split
  ✓ Ensure each class has examples in both train and test
  ✓ This requires having multiple licenses per class OR
  ✓ Grouping similar licenses into classes based on content

  Option 3: SEMANTIC SIMILARITY APPROACH
  ✓ Don't classify into exact classes
  ✓ For test license, find most similar training license
  ✓ Return that license as the "best match"
  ✓ Evaluate by similarity, not exact accuracy

NEXT STEPS:
  1. Check if raw data has multiple rows per license_id
  2. If yes: Implement proper stratified split per class
  3. If no: Use similarity-based approach (Option 3)
  4. Re-train and evaluate with proper metrics
""")
else:
    print("✓ Data structure looks reasonable for classification")

print("=" * 80)
