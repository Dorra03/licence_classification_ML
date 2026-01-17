#!/usr/bin/env python
"""Create updated feature engineering and similarity matching visualizations"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path

# Create output directory
Path('analysis').mkdir(exist_ok=True)

# ============================================================================
# VISUALIZATION 1: Updated Feature Engineering Pipeline
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Updated Feature Engineering Pipeline (5,002 Dimensions)', 
             fontsize=18, fontweight='bold', y=0.995)

# 1.1: Raw License Text Processing
ax = axes[0, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Step 1: Raw License Text', fontsize=14, fontweight='bold', pad=20)

# Input box
input_box = FancyBboxPatch((0.5, 7), 9, 2.5, boxstyle="round,pad=0.1", 
                           edgecolor='navy', facecolor='lightblue', linewidth=2)
ax.add_patch(input_box)
ax.text(5, 8.5, 'Raw License Text', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 7.8, '~100-500 words', ha='center', va='center', fontsize=10, style='italic')

# Arrow
arrow = FancyArrowPatch((5, 6.8), (5, 5.5), arrowstyle='->', mutation_scale=30, 
                       linewidth=2, color='black')
ax.add_patch(arrow)

# Process box
process_box = FancyBboxPatch((1, 3.5), 8, 1.8, boxstyle="round,pad=0.1",
                            edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
ax.add_patch(process_box)
ax.text(5, 4.6, 'Preprocessing', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 3.95, 'Lowercase • Remove punctuation • Tokenize', ha='center', va='center', fontsize=9)

# Arrow
arrow = FancyArrowPatch((5, 3.3), (5, 2.0), arrowstyle='->', mutation_scale=30,
                       linewidth=2, color='black')
ax.add_patch(arrow)

# Output box
output_box = FancyBboxPatch((0.5, 0.2), 9, 1.7, boxstyle="round,pad=0.1",
                           edgecolor='darkred', facecolor='lightyellow', linewidth=2)
ax.add_patch(output_box)
ax.text(5, 1.35, 'Cleaned Tokens', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 0.65, 'Ready for vectorization', ha='center', va='center', fontsize=9)

# 1.2: TF-IDF Vectorization
ax = axes[0, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Step 2: TF-IDF Vectorization', fontsize=14, fontweight='bold', pad=20)

# Input
input_box = FancyBboxPatch((0.5, 7.5), 9, 2, boxstyle="round,pad=0.1",
                          edgecolor='navy', facecolor='lightblue', linewidth=2)
ax.add_patch(input_box)
ax.text(5, 8.8, 'Cleaned License Tokens', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 8.0, '~50-150 unique terms', ha='center', va='center', fontsize=10, style='italic')

# Arrow
arrow = FancyArrowPatch((5, 7.3), (5, 6.0), arrowstyle='->', mutation_scale=30,
                       linewidth=2, color='black')
ax.add_patch(arrow)

# TF-IDF process
process_box = FancyBboxPatch((1, 4.2), 8, 1.6, boxstyle="round,pad=0.1",
                            edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
ax.add_patch(process_box)
ax.text(5, 5.2, 'TF-IDF Vectorizer', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 4.5, 'scikit-learn TfidfVectorizer(max_features=5000)', ha='center', va='center', fontsize=8)

# Arrow
arrow = FancyArrowPatch((5, 4.0), (5, 2.8), arrowstyle='->', mutation_scale=30,
                       linewidth=2, color='black')
ax.add_patch(arrow)

# Output boxes (show sparse matrix)
output_box = FancyBboxPatch((0.5, 0.5), 4, 2, boxstyle="round,pad=0.1",
                           edgecolor='darkred', facecolor='lightyellow', linewidth=2)
ax.add_patch(output_box)
ax.text(2.5, 1.95, 'Sparse Matrix', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(2.5, 1.3, '5,000 features', ha='center', va='center', fontsize=10, fontweight='bold', color='red')
ax.text(2.5, 0.7, '93% sparsity', ha='center', va='center', fontsize=9, style='italic')

# Info box
info_box = FancyBboxPatch((5.5, 0.5), 4, 2, boxstyle="round,pad=0.1",
                         edgecolor='purple', facecolor='lavender', linewidth=2)
ax.add_patch(info_box)
ax.text(7.5, 2.0, 'Word Importance', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(7.5, 1.4, 'Term Frequency', ha='center', va='center', fontsize=9)
ax.text(7.5, 0.95, '× Inverse Doc Freq', ha='center', va='center', fontsize=9)

# 1.3: Categorical Features
ax = axes[1, 0]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Step 3: Categorical Features', fontsize=14, fontweight='bold', pad=20)

# Input
input_box = FancyBboxPatch((0.5, 7.5), 9, 2, boxstyle="round,pad=0.1",
                          edgecolor='navy', facecolor='lightblue', linewidth=2)
ax.add_patch(input_box)
ax.text(5, 8.8, 'License Metadata', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 8.0, 'Type + OSI Status', ha='center', va='center', fontsize=10, style='italic')

# Arrow
arrow = FancyArrowPatch((5, 7.3), (5, 6.0), arrowstyle='->', mutation_scale=30,
                       linewidth=2, color='black')
ax.add_patch(arrow)

# Feature 1
feature1_box = FancyBboxPatch((0.5, 4), 4.3, 1.8, boxstyle="round,pad=0.1",
                             edgecolor='darkblue', facecolor='aliceblue', linewidth=2)
ax.add_patch(feature1_box)
ax.text(2.65, 5.3, 'License Type', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(2.65, 4.7, 'Permissive', ha='center', va='center', fontsize=8)
ax.text(2.65, 4.3, 'Copyleft (1 feature)', ha='center', va='center', fontsize=8)

# Feature 2
feature2_box = FancyBboxPatch((5.2, 4), 4.3, 1.8, boxstyle="round,pad=0.1",
                             edgecolor='darkblue', facecolor='aliceblue', linewidth=2)
ax.add_patch(feature2_box)
ax.text(7.35, 5.3, 'OSI Status', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(7.35, 4.7, 'OSI Approved', ha='center', va='center', fontsize=8)
ax.text(7.35, 4.3, 'Non-Approved (1 feature)', ha='center', va='center', fontsize=8)

# Arrow
arrow1 = FancyArrowPatch((2.65, 3.8), (3, 2.8), arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='black')
ax.add_patch(arrow1)

arrow2 = FancyArrowPatch((7.35, 3.8), (7, 2.8), arrowstyle='->', mutation_scale=20,
                        linewidth=2, color='black')
ax.add_patch(arrow2)

# Combined output
output_box = FancyBboxPatch((2, 0.5), 6, 2, boxstyle="round,pad=0.1",
                           edgecolor='darkred', facecolor='lightyellow', linewidth=2)
ax.add_patch(output_box)
ax.text(5, 1.95, '2 Categorical Features', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 1.2, 'Enhanced signal for', ha='center', va='center', fontsize=9)
ax.text(5, 0.7, 'license identification', ha='center', va='center', fontsize=9)

# 1.4: Final Feature Vector
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')
ax.set_title('Step 4: Final Feature Vector', fontsize=14, fontweight='bold', pad=20)

# TF-IDF features
tfidf_box = FancyBboxPatch((0.5, 5.5), 9, 3.5, boxstyle="round,pad=0.15",
                          edgecolor='darkgreen', facecolor='lightgreen', linewidth=3)
ax.add_patch(tfidf_box)
ax.text(5, 8.4, 'TF-IDF Features', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(5, 7.6, '5,000 word importance scores', ha='center', va='center', fontsize=11, color='darkgreen')
ax.text(5, 6.8, '[f₁, f₂, f₃, ..., f₅₀₀₀]', ha='center', va='center', fontsize=10, 
        family='monospace', style='italic')
ax.text(5, 5.9, 'Captures semantic meaning of license text', ha='center', va='center', 
        fontsize=9, style='italic')

# Categorical features
cat_box = FancyBboxPatch((0.5, 2.5), 9, 2.5, boxstyle="round,pad=0.15",
                        edgecolor='darkblue', facecolor='aliceblue', linewidth=3)
ax.add_patch(cat_box)
ax.text(5, 4.4, 'Categorical Features', ha='center', va='center', fontsize=12, fontweight='bold')
ax.text(5, 3.7, '[license_type, osi_status]', ha='center', va='center', fontsize=10, 
        family='monospace', style='italic')
ax.text(5, 3.0, '2 metadata features (0/1 encoded)', ha='center', va='center', 
        fontsize=9, style='italic')

# Final vector
final_box = FancyBboxPatch((1, 0.2), 8, 1.8, boxstyle="round,pad=0.15",
                          edgecolor='red', facecolor='mistyrose', linewidth=3)
ax.add_patch(final_box)
ax.text(5, 1.6, 'FINAL FEATURE VECTOR', ha='center', va='center', fontsize=13, fontweight='bold', color='red')
ax.text(5, 0.8, '5,002 Dimensions per License', ha='center', va='center', fontsize=12, fontweight='bold', color='darkred')

plt.tight_layout()
plt.savefig('analysis/feature_engineering_updated.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis/feature_engineering_updated.png")
plt.close()

# ============================================================================
# VISUALIZATION 2: Similarity-Based Matching Pipeline
# ============================================================================

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)

fig.suptitle('Similarity-Based Matching: From Raw Text to License Identification', 
             fontsize=18, fontweight='bold')

# Row 1: Input Processing
ax1 = fig.add_subplot(gs[0, :])
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 2)
ax1.axis('off')

# Input license text
input_box = FancyBboxPatch((0.2, 0.3), 2.5, 1.4, boxstyle="round,pad=0.1",
                          edgecolor='navy', facecolor='lightblue', linewidth=2)
ax1.add_patch(input_box)
ax1.text(1.45, 1.35, 'Unknown', ha='center', va='center', fontsize=10, fontweight='bold')
ax1.text(1.45, 0.85, 'License Text', ha='center', va='center', fontsize=9)
ax1.text(1.45, 0.45, '~200 words', ha='center', va='center', fontsize=8, style='italic')

# Arrow
arrow = FancyArrowPatch((2.8, 1), (3.5, 1), arrowstyle='->', mutation_scale=25,
                       linewidth=2, color='black')
ax1.add_patch(arrow)

# Vectorize
vec_box = FancyBboxPatch((3.5, 0.3), 2.5, 1.4, boxstyle="round,pad=0.1",
                        edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
ax1.add_patch(vec_box)
ax1.text(4.75, 1.35, 'Vectorize', ha='center', va='center', fontsize=10, fontweight='bold')
ax1.text(4.75, 0.85, 'TF-IDF', ha='center', va='center', fontsize=9)
ax1.text(4.75, 0.45, '5,000 dims', ha='center', va='center', fontsize=8, style='italic')

# Arrow
arrow = FancyArrowPatch((6.1, 1), (6.8, 1), arrowstyle='->', mutation_scale=25,
                       linewidth=2, color='black')
ax1.add_patch(arrow)

# Query vector
query_box = FancyBboxPatch((6.8, 0.3), 2.8, 1.4, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor='plum', linewidth=2)
ax1.add_patch(query_box)
ax1.text(8.2, 1.35, 'Query Vector', ha='center', va='center', fontsize=10, fontweight='bold')
ax1.text(8.2, 0.85, 'Q ∈ ℝ⁵⁰⁰⁰', ha='center', va='center', fontsize=9, family='monospace')
ax1.text(8.2, 0.45, 'Normalized', ha='center', va='center', fontsize=8, style='italic')

ax1.text(9.8, 1, '→', ha='center', va='center', fontsize=16)

# Row 2: Training Database
ax2 = fig.add_subplot(gs[1, :2])
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 5)
ax2.axis('off')
ax2.set_title('Step 2: Compare to Training Database (574 licenses)', fontsize=12, fontweight='bold', loc='left')

# Training samples visualization
colors_map = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
y_pos = 3.5
for i in range(5):
    sample_box = FancyBboxPatch((0.3 + i*1.9, y_pos), 1.7, 1.2, boxstyle="round,pad=0.05",
                               edgecolor='black', facecolor=colors_map[i], linewidth=1.5, alpha=0.7)
    ax2.add_patch(sample_box)
    ax2.text(1.15 + i*1.9, 4.2, f'License {i+1}', ha='center', va='center', fontsize=8, fontweight='bold')
    ax2.text(1.15 + i*1.9, 3.85, f'T₁ ∈ ℝ⁵⁰⁰⁰', ha='center', va='center', fontsize=7, family='monospace')

ax2.text(5, 2.7, '... + 569 more training samples ...', ha='center', va='center', fontsize=9, style='italic')

ax2.text(5, 1.8, 'All vectorized to 5,000 dimensions', ha='center', va='center', fontsize=10, fontweight='bold')
ax2.text(5, 1.2, 'Training Data: T = {T₁, T₂, ..., T₅₇₄}', ha='center', va='center', fontsize=9, family='monospace')

# Row 3: Similarity Calculation
ax3 = fig.add_subplot(gs[2, :])
ax3.set_xlim(0, 10)
ax3.set_ylim(0, 3)
ax3.axis('off')
ax3.set_title('Step 3 & 4: Calculate Similarity & Find Match', fontsize=12, fontweight='bold', loc='left')

# Cosine similarity formula
formula_box = FancyBboxPatch((0.2, 1.5), 4, 1.2, boxstyle="round,pad=0.1",
                            edgecolor='darkred', facecolor='mistyrose', linewidth=2)
ax3.add_patch(formula_box)
ax3.text(2.2, 2.35, 'Cosine Similarity', ha='center', va='center', fontsize=10, fontweight='bold')
ax3.text(2.2, 1.75, 'sim(Q, Tᵢ) = Q·Tᵢ / (||Q|| ||Tᵢ||)', ha='center', va='center', 
        fontsize=9, family='monospace')

# Arrow
arrow = FancyArrowPatch((4.3, 2), (5.2, 2), arrowstyle='->', mutation_scale=25,
                       linewidth=2, color='black')
ax3.add_patch(arrow)

# Results
result_box = FancyBboxPatch((5.2, 1.5), 4.6, 1.2, boxstyle="round,pad=0.1",
                           edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
ax3.add_patch(result_box)
ax3.text(7.5, 2.35, 'Similarity Scores', ha='center', va='center', fontsize=10, fontweight='bold')
ax3.text(7.5, 1.75, '[0.45, 0.92, 0.31, ..., 0.87]', ha='center', va='center', 
        fontsize=9, family='monospace')

# Bottom: Final result
result_final = FancyBboxPatch((1, 0.1), 8, 0.9, boxstyle="round,pad=0.1",
                             edgecolor='red', facecolor='lightyellow', linewidth=3)
ax3.add_patch(result_final)
ax3.text(5, 0.6, 'argmax(similarities) → MIT License (confidence: 0.92)', ha='center', va='center',
        fontsize=11, fontweight='bold', color='darkred')

plt.savefig('analysis/similarity_based_matching.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis/similarity_based_matching.png")
plt.close()

# ============================================================================
# VISUALIZATION 3: Complete System Architecture
# ============================================================================

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

fig.suptitle('Complete License Classification System Architecture', 
             fontsize=18, fontweight='bold', y=0.98)

# Title bar
title_bar = FancyBboxPatch((0.2, 12.8), 9.6, 0.8, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='lightgray', linewidth=2)
ax.add_patch(title_bar)
ax.text(5, 13.2, 'Automated License Classification Pipeline', ha='center', va='center',
       fontsize=12, fontweight='bold')

# INPUT LAYER
input_box = FancyBboxPatch((0.5, 11.5), 9, 1, boxstyle="round,pad=0.1",
                          edgecolor='navy', facecolor='lightblue', linewidth=2)
ax.add_patch(input_box)
ax.text(5, 12.15, 'INPUT: Raw License Text Files', ha='center', va='center', fontsize=11, fontweight='bold')
ax.text(5, 11.7, 'Web UI  •  CLI  •  REST API  •  Batch Processing', ha='center', va='center', fontsize=9)

# PREPROCESSING
arrow = FancyArrowPatch((5, 11.4), (5, 10.8), arrowstyle='->', mutation_scale=30,
                       linewidth=2, color='black')
ax.add_patch(arrow)

preprocess_box = FancyBboxPatch((0.8, 9.8), 8.4, 0.9, boxstyle="round,pad=0.1",
                               edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
ax.add_patch(preprocess_box)
ax.text(5, 10.45, 'PREPROCESSING: Tokenize, lowercase, clean', ha='center', va='center', fontsize=10, fontweight='bold')

# FEATURE ENGINEERING (TF-IDF)
arrow = FancyArrowPatch((5, 9.7), (5, 9.1), arrowstyle='->', mutation_scale=30,
                       linewidth=2, color='black')
ax.add_patch(arrow)

tfidf_box = FancyBboxPatch((0.5, 8), 4.3, 1.1, boxstyle="round,pad=0.1",
                          edgecolor='darkblue', facecolor='aliceblue', linewidth=2)
ax.add_patch(tfidf_box)
ax.text(2.65, 8.8, 'TF-IDF Vectorization', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(2.65, 8.35, '5,000 word features', ha='center', va='center', fontsize=9)

# CATEGORICAL FEATURES
cat_box = FancyBboxPatch((5.2, 8), 4.3, 1.1, boxstyle="round,pad=0.1",
                        edgecolor='darkblue', facecolor='aliceblue', linewidth=2)
ax.add_patch(cat_box)
ax.text(7.35, 8.8, 'Categorical Features', ha='center', va='center', fontsize=10, fontweight='bold')
ax.text(7.35, 8.35, 'Type + OSI status (2)', ha='center', va='center', fontsize=9)

# Arrows to combined
arrow1 = FancyArrowPatch((2.65, 7.9), (4, 7.3), arrowstyle='->', mutation_scale=25,
                        linewidth=2, color='black')
ax.add_patch(arrow1)

arrow2 = FancyArrowPatch((7.35, 7.9), (6, 7.3), arrowstyle='->', mutation_scale=25,
                        linewidth=2, color='black')
ax.add_patch(arrow2)

# COMBINED FEATURE VECTOR
feature_box = FancyBboxPatch((2, 5.8), 6, 1.4, boxstyle="round,pad=0.1",
                            edgecolor='purple', facecolor='plum', linewidth=3)
ax.add_patch(feature_box)
ax.text(5, 6.85, 'FEATURE VECTOR: 5,002 Dimensions', ha='center', va='center', 
       fontsize=11, fontweight='bold', color='darkviolet')
ax.text(5, 6.3, '[f₁, f₂, ..., f₅₀₀₀, type, osi_status]', ha='center', va='center', 
       fontsize=9, family='monospace')

# MODELS (parallel)
arrow = FancyArrowPatch((5, 5.7), (5, 5.2), arrowstyle='->', mutation_scale=30,
                       linewidth=2, color='black')
ax.add_patch(arrow)

# RF Model
rf_box = FancyBboxPatch((0.5, 3.5), 2.2, 1.6, boxstyle="round,pad=0.1",
                       edgecolor='darkred', facecolor='mistyrose', linewidth=2)
ax.add_patch(rf_box)
ax.text(1.6, 4.8, 'Random Forest', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(1.6, 4.3, 'Accuracy:', ha='center', va='center', fontsize=8)
ax.text(1.6, 3.95, '91.8%', ha='center', va='center', fontsize=10, fontweight='bold', color='red')
ax.text(1.6, 3.65, '(Best Model)', ha='center', va='center', fontsize=7, style='italic')

# NB Model
nb_box = FancyBboxPatch((2.9, 3.5), 2.2, 1.6, boxstyle="round,pad=0.1",
                       edgecolor='darkblue', facecolor='aliceblue', linewidth=2)
ax.add_patch(nb_box)
ax.text(4, 4.8, 'Naive Bayes', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(4, 4.3, 'Accuracy:', ha='center', va='center', fontsize=8)
ax.text(4, 3.95, '83.8%', ha='center', va='center', fontsize=10, fontweight='bold', color='blue')
ax.text(4, 3.65, '(Fast)', ha='center', va='center', fontsize=7, style='italic')

# ANN Model
ann_box = FancyBboxPatch((5.3, 3.5), 2.2, 1.6, boxstyle="round,pad=0.1",
                        edgecolor='darkgreen', facecolor='lightgreen', linewidth=2)
ax.add_patch(ann_box)
ax.text(6.4, 4.8, 'Neural Network', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(6.4, 4.3, 'Accuracy:', ha='center', va='center', fontsize=8)
ax.text(6.4, 3.95, '45.5%', ha='center', va='center', fontsize=10, fontweight='bold', color='green')
ax.text(6.4, 3.65, '(Low)', ha='center', va='center', fontsize=7, style='italic')

# CNN Model
cnn_box = FancyBboxPatch((7.7, 3.5), 2.2, 1.6, boxstyle="round,pad=0.1",
                        edgecolor='orange', facecolor='moccasin', linewidth=2)
ax.add_patch(cnn_box)
ax.text(8.8, 4.8, 'CNN', ha='center', va='center', fontsize=9, fontweight='bold')
ax.text(8.8, 4.3, 'Accuracy:', ha='center', va='center', fontsize=8)
ax.text(8.8, 3.95, '80.0%', ha='center', va='center', fontsize=10, fontweight='bold', color='darkorange')
ax.text(8.8, 3.65, '(Good)', ha='center', va='center', fontsize=7, style='italic')

# Arrows from models to matching
for x_pos in [1.6, 4, 6.4, 8.8]:
    arrow = FancyArrowPatch((x_pos, 3.4), (5, 2.8), arrowstyle='->', mutation_scale=20,
                           linewidth=1.5, color='gray', alpha=0.6)
    ax.add_patch(arrow)

# SIMILARITY MATCHING
match_box = FancyBboxPatch((1.5, 1.5), 7, 1.2, boxstyle="round,pad=0.1",
                          edgecolor='red', facecolor='lightyellow', linewidth=3)
ax.add_patch(match_box)
ax.text(5, 2.35, 'SIMILARITY-BASED MATCHING', ha='center', va='center', 
       fontsize=11, fontweight='bold', color='darkred')
ax.text(5, 1.85, 'Cosine Similarity vs 574 Training Licenses  |  Find Best Match (0.87 avg)', 
       ha='center', va='center', fontsize=9)

# OUTPUT
arrow = FancyArrowPatch((5, 1.4), (5, 0.8), arrowstyle='->', mutation_scale=30,
                       linewidth=2, color='black')
ax.add_patch(arrow)

output_box = FancyBboxPatch((1, 0), 8, 0.7, boxstyle="round,pad=0.1",
                           edgecolor='darkgreen', facecolor='lightgreen', linewidth=3)
ax.add_patch(output_box)
ax.text(5, 0.5, 'OUTPUT: SPDX License ID  +  Confidence Score  +  All 4 Model Predictions', 
       ha='center', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('analysis/system_architecture_complete.png', dpi=300, bbox_inches='tight')
print("✓ Saved: analysis/system_architecture_complete.png")
plt.close()

print()
print("=" * 80)
print("ALL VISUALIZATIONS CREATED SUCCESSFULLY")
print("=" * 80)
print()
print("Files created:")
print("  1. analysis/feature_engineering_updated.png - Feature pipeline (5,002 dims)")
print("  2. analysis/similarity_based_matching.png - Matching algorithm visualization")
print("  3. analysis/system_architecture_complete.png - Complete system overview")
print()
print("These visualizations show:")
print("  ✓ How raw text becomes 5,002-dimensional vectors")
print("  ✓ TF-IDF (5,000) + Categorical (2) feature composition")
print("  ✓ Similarity-based matching instead of classification")
print("  ✓ Complete end-to-end system architecture")
print()
