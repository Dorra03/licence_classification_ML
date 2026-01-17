#!/usr/bin/env python
"""
Generate X_train_fixed.npz and X_test_fixed.npz from existing y_train_fixed.csv and y_test_fixed.csv
This extracts features from the XML files referenced in the CSV labels
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xml.etree.ElementTree as ET
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import save_npz, hstack, csr_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("REGENERATING FEATURE MATRICES FROM LABELS")
print("=" * 80)

# Load label files
y_train_df = pd.read_csv('data/features/y_train_fixed.csv')
y_test_df = pd.read_csv('data/features/y_test_fixed.csv')

print(f"\n✓ Loaded training labels: {len(y_train_df)} samples")
print(f"✓ Loaded test labels: {len(y_test_df)} samples")

# Function to extract text from XML license
def extract_license_text(license_id):
    """Extract text content from SPDX license XML file"""
    xml_path = Path('data/raw/license-list-XML') / f'{license_id}.xml'
    
    if not xml_path.exists():
        return ""
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract all text content
        text_parts = []
        for elem in root.iter():
            if elem.text:
                text_parts.append(elem.text.strip())
        
        return ' '.join(text_parts)
    except Exception as e:
        print(f"  Warning: Could not parse {license_id}: {e}")
        return ""

# Extract text for all licenses
print("\n[1] EXTRACTING LICENSE TEXT FROM XML FILES")
print("-" * 80)

all_licenses = list(y_train_df['license_id'].unique()) + list(y_test_df['license_id'].unique())
license_texts = {}

print(f"Extracting text for {len(all_licenses)} unique licenses...")
for i, license_id in enumerate(all_licenses):
    if (i + 1) % 100 == 0:
        print(f"  Progress: {i + 1} / {len(all_licenses)}")
    
    license_texts[license_id] = extract_license_text(license_id)

print(f"✓ Extracted text for {len(license_texts)} licenses")

# Prepare training data
print("\n[2] PREPARING FEATURE VECTORS")
print("-" * 80)

X_train_texts = [license_texts[lid] for lid in y_train_df['license_id']]
X_test_texts = [license_texts[lid] for lid in y_test_df['license_id']]

print(f"Training texts: {len(X_train_texts)}")
print(f"Test texts: {len(X_test_texts)}")

# TF-IDF Vectorization
print("\nApplying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)

X_train_tfidf = vectorizer.fit_transform(X_train_texts)
X_test_tfidf = vectorizer.transform(X_test_texts)

print(f"✓ TF-IDF shape: {X_train_tfidf.shape}")
print(f"✓ Sparsity: {100 * (1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])):.2f}%")

# Add categorical features (text length, word count)
print("\nAdding categorical features...")

def get_categorical_features(texts):
    """Extract length and word count features"""
    features = []
    for text in texts:
        word_count = len(text.split())
        char_count = len(text)
        features.append([word_count, char_count])
    return np.array(features, dtype=np.float32)

X_train_cat = get_categorical_features(X_train_texts)
X_test_cat = get_categorical_features(X_test_texts)

# Normalize categorical features
X_train_cat = X_train_cat / (np.max(X_train_cat, axis=0) + 1e-8)
X_test_cat = X_test_cat / (np.max(X_test_cat, axis=0) + 1e-8)

# Combine TF-IDF and categorical
X_train_cat_sparse = csr_matrix(X_train_cat)
X_test_cat_sparse = csr_matrix(X_test_cat)

X_train_full = hstack([X_train_tfidf, X_train_cat_sparse])
X_test_full = hstack([X_test_tfidf, X_test_cat_sparse])

print(f"✓ Combined features shape:")
print(f"  X_train: {X_train_full.shape}")
print(f"  X_test: {X_test_full.shape}")

# Save NPZ files
print("\n[3] SAVING FEATURE MATRICES")
print("-" * 80)

save_npz('data/features/X_train_fixed.npz', X_train_full)
save_npz('data/features/X_test_fixed.npz', X_test_full)
print("✓ Saved X_train_fixed.npz")
print("✓ Saved X_test_fixed.npz")

# Save vectorizer
with open('data/features/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✓ Saved vectorizer.pkl")

print("\n" + "=" * 80)
print("FEATURE GENERATION COMPLETE")
print("=" * 80)
print(f"\nFiles ready for training:")
print(f"  - data/features/X_train_fixed.npz ({X_train_full.shape})")
print(f"  - data/features/X_test_fixed.npz ({X_test_full.shape})")
print(f"  - data/features/y_train_fixed.csv ({len(y_train_df)} labels)")
print(f"  - data/features/y_test_fixed.csv ({len(y_test_df)} labels)")
