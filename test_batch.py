#!/usr/bin/env python
"""Test batch_classifier with actual XML files"""

from batch_classifier import LicenseClassifier
from pathlib import Path

print("Testing batch_classifier with XML files...")
print()

lc = LicenseClassifier()
results = lc.classify_directory('data/raw/license-list-XML', '*.xml')

print(f"Processed {len(results)} files")
print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
print()

print("First 10 results:")
print("-" * 100)
for r in results[:10]:
    status = "[OK]" if r['status'] == 'success' else "[ERROR]"
    license_id = r.get('license_id') or 'NONE'
    confidence = r.get('confidence') or 0.0
    print(f"{status} {r['filename']:40s} → {license_id:25s} ({confidence:.1%})")
print()

# Calculate accuracy
correct = sum(1 for r in results if r['status'] == 'success' 
              and r['license_id'].replace('.xml', '') == Path(r['filename']).stem)
print(f"Accuracy: {correct}/{len(results)} ({100*correct/len(results):.1f}%)")
