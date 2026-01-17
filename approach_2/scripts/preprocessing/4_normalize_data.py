"""
Phase 2, Step 4: Normalize and Prepare Data
- Load unified dataset from Phase 1
- Normalize text data
- Clean and standardize
- Prepare for feature engineering
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import re
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def load_unified_schema():
    """Load the unified label schema and license texts from Phase 1"""
    logger.info("Loading unified label schema and license texts...")
    
    schema_file = DATA_DIR / "processed" / "unified_label_schema.json"
    license_texts_file = DATA_DIR / "spdx" / "license_texts.json"
    
    if not schema_file.exists():
        logger.error(f"Unified schema not found at {schema_file}")
        logger.error("Please run Phase 1 scripts first!")
        return None
    
    with open(schema_file) as f:
        schema = json.load(f)
    
    # Load license texts
    license_texts = {}
    if license_texts_file.exists():
        with open(license_texts_file) as f:
            license_texts = json.load(f)
    
    logger.info(f"Loaded schema with {len(schema.get('classes', {}))} licenses")
    logger.info(f"Loaded texts for {len(license_texts)} licenses")
    
    # Combine schema with texts
    combined = {
        'classes': schema.get('classes', {}),
        'texts': license_texts,
        'stats': schema.get('summary', {})
    }
    
    return combined

def normalize_text(text):
    """Normalize license text"""
    if not text or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove non-ASCII characters but keep alphanumeric and common punctuation
    text = re.sub(r'[^\w\s\-.,;:()"\']', '', text)
    
    return text

def create_dataset():
    """Create a normalized dataset"""
    logger.info("Creating normalized dataset...")
    
    combined = load_unified_schema()
    if combined is None:
        return None
    
    schema = combined['classes']
    texts = combined['texts']
    
    # Create dataframe
    data = {
        'license_id': [],
        'license_name': [],
        'text': [],
        'category': [],
        'source': [],
        'aliases': []
    }
    
    for license_id, license_info in schema.items():
        # Get text from license_texts or use empty string
        text = texts.get(license_id, '')
        
        data['license_id'].append(license_id)
        data['license_name'].append(license_info.get('name', ''))
        data['text'].append(normalize_text(text))
        
        # Get first category if available
        categories = license_info.get('categories', ['other'])
        category = categories[0] if categories else 'other'
        data['category'].append(category)
        
        data['source'].append(license_info.get('source', 'spdx'))
        data['aliases'].append(license_info.get('aliases', []))
    
    df = pd.DataFrame(data)
    
    logger.info(f"Created dataset with {len(df)} licenses")
    
    # Check that text column is strings
    if len(df) > 0 and df['text'].dtype == 'object':
        logger.info(f"Text columns with content: {(df['text'].str.len() > 0).sum()}")
        logger.info(f"Categories: {dict(df['category'].value_counts())}")
    
    return df

def remove_duplicates(df):
    """Remove duplicate texts"""
    logger.info("Removing duplicates...")
    
    before = len(df)
    
    # Remove exact duplicates based on text
    df = df.drop_duplicates(subset=['text'], keep='first')
    
    after = len(df)
    logger.info(f"Removed {before - after} duplicates, {after} licenses remaining")
    
    return df

def filter_empty(df):
    """Remove licenses with empty text"""
    logger.info("Filtering empty texts...")
    
    before = len(df)
    df = df[df['text'].str.len() > 50]  # Keep only if text is meaningful
    after = len(df)
    
    logger.info(f"Removed {before - after} empty/short texts, {after} licenses remaining")
    
    return df

def save_normalized_data(df):
    """Save normalized data"""
    logger.info("Saving normalized data...")
    
    # Save as CSV
    csv_file = PROCESSED_DIR / "normalized_dataset.csv"
    df.to_csv(csv_file, index=False)
    logger.info(f"Saved to {csv_file}")
    
    # Save as JSON
    json_file = PROCESSED_DIR / "normalized_dataset.json"
    with open(json_file, 'w') as f:
        json.dump(df.to_dict(orient='records'), f, indent=2)
    logger.info(f"Saved to {json_file}")
    
    # Save statistics
    stats = {
        'total_licenses': int(len(df)),
        'categories': {k: int(v) for k, v in df['category'].value_counts().to_dict().items()},
        'sources': {k: int(v) for k, v in df['source'].value_counts().to_dict().items()},
        'avg_text_length': float(df['text'].str.len().mean()),
        'min_text_length': int(df['text'].str.len().min()),
        'max_text_length': int(df['text'].str.len().max())
    }
    
    stats_file = PROCESSED_DIR / "normalization_statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_file}")
    
    return df

def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("PHASE 2, STEP 4: NORMALIZE DATA")
    logger.info("=" * 80)
    
    try:
        # Load data
        df = create_dataset()
        if df is None:
            logger.error("Failed to create dataset")
            return False
        
        # Clean data
        df = filter_empty(df)
        df = remove_duplicates(df)
        
        # Save
        df = save_normalized_data(df)
        
        logger.info("=" * 80)
        logger.info("✓ PHASE 2, STEP 4 COMPLETE")
        logger.info(f"  Final dataset: {len(df)} licenses")
        logger.info(f"  Saved to: {PROCESSED_DIR}")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in normalization: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
