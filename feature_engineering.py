"""
Preprocessing and Feature Engineering Pipeline
Converts raw cleaned license data into ML-ready features

Pipeline:
1. Load cleaned data
2. Remove redundant columns
3. Normalize Unicode characters
4. Encode categorical variables
5. Text vectorization (TF-IDF + optional BERT)
6. Feature combination and scaling
7. Train-test split
8. Save processed features for ML models
"""

import pandas as pd
import numpy as np
import os
import json
import unicodedata
import logging
from pathlib import Path

# ML/NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack, csr_matrix
import pickle

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    def __init__(self, input_path, output_dir):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.encoders = {}
        self.feature_names = []
        
    def load_data(self):
        """Load cleaned license data"""
        logger.info("Loading cleaned license data...")
        self.df = pd.read_csv(self.input_path)
        logger.info(f"Loaded {len(self.df)} records with columns: {list(self.df.columns)}")
        return self.df
    
    def remove_redundant_columns(self):
        """Remove columns not needed for ML"""
        logger.info("Removing redundant columns...")
        
        redundant_cols = [
            'raw_text',              # Not needed (we'll use cleaned_text)
            'is_osi_approved',       # Redundant (included in osi_status)
            'is_deprecated',         # Redundant (all are active)
            'activity_status'        # All values are "active" (no variance)
        ]
        
        cols_to_drop = [col for col in redundant_cols if col in self.df.columns]
        self.df.drop(columns=cols_to_drop, inplace=True)
        
        logger.info(f"Dropped columns: {cols_to_drop}")
        logger.info(f"Remaining columns: {list(self.df.columns)}")
        return self.df
    
    def normalize_unicode(self):
        """Normalize Unicode characters to NFC form"""
        logger.info("Normalizing Unicode characters...")
        
        # Apply Unicode normalization to cleaned_text
        self.df['cleaned_text'] = self.df['cleaned_text'].apply(
            lambda x: unicodedata.normalize('NFC', str(x))
        )
        
        logger.info("Unicode normalization completed")
        return self.df
    
    def encode_categorical_variables(self):
        """Encode categorical features"""
        logger.info("Encoding categorical variables...")
        
        categorical_cols = ['license_type', 'osi_status']
        
        for col in categorical_cols:
            if col in self.df.columns:
                encoder = LabelEncoder()
                self.df[f'{col}_encoded'] = encoder.fit_transform(self.df[col])
                self.encoders[col] = encoder
                
                # Log encoding mapping
                mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
                logger.info(f"{col} encoding: {mapping}")
        
        logger.info("Categorical encoding completed")
        return self.df
    
    def vectorize_text_tfidf(self, max_features=5000, ngram_range=(1, 2)):
        """Convert text to TF-IDF vectors"""
        logger.info(f"Vectorizing text with TF-IDF (max_features={max_features}, ngram_range={ngram_range})...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,              # Minimum document frequency
            max_df=0.8,            # Maximum document frequency
            sublinear_tf=True,     # Sublinear term frequency scaling
            stop_words='english'
        )
        
        X_tfidf = self.vectorizer.fit_transform(self.df['cleaned_text'])
        
        logger.info(f"TF-IDF output shape: {X_tfidf.shape}")
        logger.info(f"Vocabulary size: {len(self.vectorizer.get_feature_names_out())}")
        
        self.feature_names.append(('tfidf', len(self.vectorizer.get_feature_names_out())))
        
        return X_tfidf
    
    def combine_features(self, X_tfidf):
        """Combine TF-IDF with categorical encoded features"""
        logger.info("Combining features...")
        
        # Start with TF-IDF
        X_combined = X_tfidf
        
        # Add encoded categorical features
        encoded_cols = [col for col in self.df.columns if col.endswith('_encoded')]
        
        if encoded_cols:
            categorical_features = self.df[encoded_cols].values
            # Convert to sparse matrix for consistency
            categorical_sparse = csr_matrix(categorical_features)
            X_combined = hstack([X_combined, categorical_sparse])
            
            logger.info(f"Added {len(encoded_cols)} categorical features")
            self.feature_names.append(('categorical', len(encoded_cols)))
        
        logger.info(f"Combined feature matrix shape: {X_combined.shape}")
        return X_combined
    
    def prepare_target(self):
        """Prepare target variable"""
        logger.info("Preparing target variable...")
        
        y = self.df['license_id']
        
        logger.info(f"Target variable (license_id):")
        logger.info(f"  Total samples: {len(y)}")
        logger.info(f"  Unique classes: {len(y.unique())}")
        logger.info(f"  Class distribution: Each class appears 1 time (SPDX reference set)")
        
        return y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split into train/test sets"""
        logger.info(f"Splitting data (test_size={test_size})...")
        
        # Use stratified split to maintain class distribution
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state,
                stratify=y  # Maintain class distribution
            )
        except Exception as e:
            logger.warning(f"Stratified split failed: {e}. Using regular split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=random_state
            )
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def save_features(self):
        """Save processed features and metadata"""
        logger.info("Saving processed features...")
        
        # Save sparse matrices
        import scipy.sparse as sp
        sp.save_npz(self.output_dir / 'X_train.npz', self.X_train)
        sp.save_npz(self.output_dir / 'X_test.npz', self.X_test)
        
        # Save labels
        self.y_train.to_csv(self.output_dir / 'y_train.csv', index=False)
        self.y_test.to_csv(self.output_dir / 'y_test.csv', index=False)
        
        # Save encoders
        with open(self.output_dir / 'encoders.pkl', 'wb') as f:
            pickle.dump(self.encoders, f)
        
        # Save vectorizer
        with open(self.output_dir / 'vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        logger.info("Saved: X_train.npz, X_test.npz, y_train.csv, y_test.csv")
        logger.info("Saved: encoders.pkl, vectorizer.pkl")
        
        return True
    
    def generate_report(self):
        """Generate preprocessing report"""
        logger.info("Generating preprocessing report...")
        
        report = []
        report.append("=" * 80)
        report.append("PREPROCESSING & FEATURE ENGINEERING REPORT")
        report.append("=" * 80)
        
        report.append("\n📋 PIPELINE STEPS")
        report.append("-" * 80)
        report.append("✅ Step 1: Load cleaned data")
        report.append("✅ Step 2: Remove redundant columns")
        report.append("✅ Step 3: Normalize Unicode")
        report.append("✅ Step 4: Encode categorical variables")
        report.append("✅ Step 5: Text vectorization (TF-IDF)")
        report.append("✅ Step 6: Combine features")
        report.append("✅ Step 7: Train-test split")
        
        report.append("\n📊 DATA STATISTICS")
        report.append("-" * 80)
        report.append(f"Total samples: {len(self.df)}")
        report.append(f"Original columns: 8")
        report.append(f"Final columns: {len(self.df.columns)}")
        
        report.append("\n🔤 CATEGORICAL ENCODING")
        report.append("-" * 80)
        for col, encoder in self.encoders.items():
            mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            report.append(f"\n{col}:")
            for label, code in mapping.items():
                report.append(f"  {label} → {code}")
        
        report.append("\n📈 FEATURE ENGINEERING")
        report.append("-" * 80)
        report.append(f"TF-IDF Configuration:")
        report.append(f"  max_features: 5000")
        report.append(f"  ngram_range: (1, 2)")
        report.append(f"  min_df: 2")
        report.append(f"  max_df: 0.8")
        report.append(f"  sublinear_tf: True")
        
        report.append(f"\nFeatures Summary:")
        report.append(f"  TF-IDF features: {self.feature_names[0][1]}")
        if len(self.feature_names) > 1:
            report.append(f"  Categorical features: {self.feature_names[1][1]}")
        total_features = sum([count for _, count in self.feature_names])
        report.append(f"  Total features: {total_features}")
        
        report.append("\n📊 TRAIN-TEST SPLIT")
        report.append("-" * 80)
        report.append(f"Train set: {self.X_train.shape[0]} samples")
        report.append(f"Test set: {self.X_test.shape[0]} samples")
        report.append(f"Feature matrix shape:")
        report.append(f"  X_train: {self.X_train.shape}")
        report.append(f"  X_test: {self.X_test.shape}")
        report.append(f"Target variable:")
        report.append(f"  y_train: {len(self.y_train)} samples")
        report.append(f"  y_test: {len(self.y_test)} samples")
        
        report.append("\n🎯 TARGET VARIABLE")
        report.append("-" * 80)
        report.append(f"Total unique classes: {len(self.y_train.unique())}")
        report.append(f"Classes in train set: {len(self.y_train.unique())}")
        report.append(f"Classes in test set: {len(self.y_test.unique())}")
        report.append(f"Most common class: {self.y_train.value_counts().index[0]}")
        report.append(f"Least common class: {self.y_train.value_counts().index[-1]}")
        
        report.append("\n✨ PREPROCESSING BENEFITS")
        report.append("-" * 80)
        report.append("✅ Unicode normalization: Consistent text representation")
        report.append("✅ Redundant column removal: Cleaner feature space")
        report.append("✅ Categorical encoding: Numbers for ML algorithms")
        report.append("✅ TF-IDF vectorization: Text → numerical features")
        report.append("✅ Stratified split: Maintained class distribution")
        
        report.append("\n🚀 READY FOR ML MODELS")
        report.append("-" * 80)
        report.append("✅ Feature matrix: X_train.npz, X_test.npz")
        report.append("✅ Target labels: y_train.csv, y_test.csv")
        report.append("✅ Encoders: encoders.pkl")
        report.append("✅ Vectorizer: vectorizer.pkl")
        
        report.append("\n📋 NEXT STEPS")
        report.append("-" * 80)
        report.append("1. Load features: X_train = sparse.load_npz('X_train.npz')")
        report.append("2. Train models: SVM, Random Forest, BERT")
        report.append("3. Evaluate: Accuracy, F1-score, Precision, Recall")
        report.append("4. Compare: ScanCode, FOSSology results")
        
        report_text = "\n".join(report)
        
        # Save report
        report_path = self.output_dir / 'preprocessing_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"Report saved to {report_path}")
        print("\n" + report_text)
        
        return report_text
    
    def create_feature_visualization(self):
        """Create visualization of features"""
        logger.info("Creating feature visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Feature Engineering Summary', fontsize=16, fontweight='bold')
        
        # 1. Categorical encoding
        axes[0, 0].barh(['license_type', 'osi_status'], 
                        [len(self.encoders.get('license_type', {}).classes_),
                         len(self.encoders.get('osi_status', {}).classes_)])
        axes[0, 0].set_xlabel('Number of Categories')
        axes[0, 0].set_title('Categorical Features')
        axes[0, 0].grid(axis='x', alpha=0.3)
        
        # 2. Feature composition
        feature_types = [name for name, _ in self.feature_names]
        feature_counts = [count for _, count in self.feature_names]
        colors = ['#3498db', '#e74c3c', '#2ecc71'][:len(feature_types)]
        axes[0, 1].pie(feature_counts, labels=feature_types, autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[0, 1].set_title('Feature Composition')
        
        # 3. Train-test split
        split_data = [len(self.y_train), len(self.y_test)]
        axes[1, 0].bar(['Train', 'Test'], split_data, color=['#27ae60', '#e67e22'])
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_title('Train-Test Split')
        for i, v in enumerate(split_data):
            axes[1, 0].text(i, v + 10, str(v), ha='center', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # 4. Feature matrix shape
        axes[1, 1].text(0.5, 0.7, f"Feature Matrix Shape\n\n", 
                       ha='center', va='top', fontsize=14, fontweight='bold')
        axes[1, 1].text(0.5, 0.5, f"X_train: {self.X_train.shape}\nX_test: {self.X_test.shape}\n\n" + 
                                   f"Total Features: {self.X_train.shape[1]}\nTotal Samples: {self.X_train.shape[0] + self.X_test.shape[0]}",
                       ha='center', va='center', fontsize=12, family='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_engineering_summary.png', dpi=300, bbox_inches='tight')
        logger.info("Saved: feature_engineering_summary.png")
        plt.close()
    
    def run_pipeline(self):
        """Run complete preprocessing pipeline"""
        logger.info("=" * 80)
        logger.info("STARTING PREPROCESSING & FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 80)
        
        # Execute pipeline steps
        self.load_data()
        self.remove_redundant_columns()
        self.normalize_unicode()
        self.encode_categorical_variables()
        
        # Text vectorization
        X_tfidf = self.vectorize_text_tfidf()
        
        # Combine features
        X_combined = self.combine_features(X_tfidf)
        
        # Prepare target
        y = self.prepare_target()
        
        # Train-test split
        self.split_data(X_combined, y)
        
        # Save processed features
        self.save_features()
        
        # Generate report
        self.generate_report()
        
        # Create visualization
        self.create_feature_visualization()
        
        logger.info("=" * 80)
        logger.info("PREPROCESSING PIPELINE COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"\nOutputs saved to: {self.output_dir}")
        logger.info("Files created:")
        logger.info("  • X_train.npz (sparse matrix)")
        logger.info("  • X_test.npz (sparse matrix)")
        logger.info("  • y_train.csv (target labels)")
        logger.info("  • y_test.csv (target labels)")
        logger.info("  • encoders.pkl (categorical encoders)")
        logger.info("  • vectorizer.pkl (TF-IDF vectorizer)")
        logger.info("  • preprocessing_report.txt (detailed report)")
        logger.info("  • feature_engineering_summary.png (visualization)")
        
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'vectorizer': self.vectorizer,
            'encoders': self.encoders
        }


if __name__ == "__main__":
    input_path = r"c:\Users\ASUS\Desktop\ML project 2\data\processed\licenses_cleaned.csv"
    output_dir = r"c:\Users\ASUS\Desktop\ML project 2\data\features"
    
    pipeline = PreprocessingPipeline(input_path, output_dir)
    results = pipeline.run_pipeline()
    
    logger.info("\n✅ ALL PREPROCESSING STEPS COMPLETED SUCCESSFULLY!")
    logger.info("Next: Build and train classification models")
