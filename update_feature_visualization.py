"""
Update Feature Engineering Summary Visualization
Shows correct feature composition with categorical features
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Feature Engineering Summary', fontsize=16, fontweight='bold')

# 1. Categorical Features
categorical_features = {
    'license_type': 3,
    'osi_status': 3
}
colors_cat = ['#3498db', '#e74c3c']
axes[0, 0].barh(list(categorical_features.keys()), list(categorical_features.values()), color=colors_cat)
axes[0, 0].set_xlabel('Number of Categories', fontweight='bold')
axes[0, 0].set_title('Categorical Features', fontsize=12, fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)
for i, (key, v) in enumerate(categorical_features.items()):
    axes[0, 0].text(v + 0.2, i, str(v), va='center', fontweight='bold', fontsize=11)

# 2. Feature Composition (Pie Chart)
feature_composition = {
    'TF-IDF (5000)': 5000,
    'Categorical (2)': 2
}
colors_pie = ['#3498db', '#e74c3c']
wedges, texts, autotexts = axes[0, 1].pie(
    feature_composition.values(), 
    labels=feature_composition.keys(), 
    autopct='%1.2f%%',
    colors=colors_pie, 
    startangle=90,
    textprops={'fontsize': 11, 'weight': 'bold'}
)
axes[0, 1].set_title('Feature Composition (5,002 Total)', fontsize=12, fontweight='bold')

# Make percentage text more visible
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
    autotext.set_weight('bold')

# 3. Train-Test Split
split_data = [574, 144]
split_labels = ['Train', 'Test']
colors_split = ['#27ae60', '#e67e22']
bars = axes[1, 0].bar(split_labels, split_data, color=colors_split, edgecolor='black', linewidth=1.5)
axes[1, 0].set_ylabel('Number of Samples', fontweight='bold', fontsize=11)
axes[1, 0].set_title('Train-Test Split', fontsize=12, fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, v in zip(bars, split_data):
    height = bar.get_height()
    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(v)}\n({v/sum(split_data)*100:.1f}%)',
                   ha='center', va='bottom', fontweight='bold', fontsize=10)

# 4. Feature Matrix Summary
summary_text = """Feature Matrix Summary

Shape Information:
  • X_train: (574, 5002)
  • X_test: (144, 5002)

Feature Details:
  • Total Features: 5,002
  • Total Samples: 718
  • Feature Sparsity: 92.94%

Feature Breakdown:
  • TF-IDF: 5,000 (99.96%)
  • Categorical: 2 (0.04%)
    - license_type (3 categories)
    - osi_status (3 categories)
      
License Type Distribution:
  • other: 567 (79.0%)
  • copyleft: 78 (10.9%)
  • permissive: 73 (10.2%)
"""

axes[1, 1].text(0.05, 0.95, summary_text, 
               ha='left', va='top', fontsize=10, family='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2, pad=1),
               transform=axes[1, 1].transAxes)
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('data/features/feature_engineering_summary.png', dpi=300, bbox_inches='tight')
print("✓ Updated: feature_engineering_summary.png")
print("  - Shows 99.96% TF-IDF + 0.04% Categorical")
print("  - Categorical features:")
print("    - license_type: 3 categories (other, copyleft, permissive)")
print("    - osi_status: 3 categories")
print("  - Total features: 5,002")
print("  - Train samples: 574 | Test samples: 144")
plt.close()
