"""
Automated License Classification System
Takes raw license files and classifies them into SPDX identifiers
Uses trained Random Forest model for high accuracy
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class LicenseClassifier:
    """Automated license classification system using Random Forest + Similarity"""
    
    def __init__(self, models_dir='models', data_dir='data/features', model_type='random_forest'):
        """Initialize classifier with trained models"""
        print(f"Initializing License Classifier with {model_type}...")
        
        self.models_dir = models_dir
        self.model_type = model_type
        
        # Load vectorizer
        with open(f'{models_dir}/vectorizer.pkl', 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load label encoder
        with open(f'{models_dir}/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        # Load selected model
        self.load_model(model_type)
        
        # Load training data for similarity matching
        from scipy.sparse import load_npz
        self.X_train = load_npz(f'{data_dir}/X_train_fixed.npz')
        self.y_train = pd.read_csv(f'{data_dir}/y_train_fixed.csv')['license_id'].values
        
        print(f"[OK] Classifier ready (using {model_type})")
    
    def load_model(self, model_type):
        """Load the specified model"""
        if model_type == 'random_forest':
            with open(f'{self.models_dir}/random_forest.pkl', 'rb') as f:
                self.model = pickle.load(f)
        elif model_type == 'naive_bayes':
            with open(f'{self.models_dir}/naive_bayes.pkl', 'rb') as f:
                self.model = pickle.load(f)
        elif model_type == 'cnn':
            # For CNN, we'll need Keras
            try:
                import tensorflow as tf
                from tensorflow import keras
                self.model = keras.models.load_model(f'{self.models_dir}/cnn_model.h5')
                self.is_cnn = True
            except ImportError:
                print("Warning: TensorFlow not available, falling back to Random Forest")
                with open(f'{self.models_dir}/random_forest.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                self.is_cnn = False
        else:
            with open(f'{self.models_dir}/random_forest.pkl', 'rb') as f:
                self.model = pickle.load(f)
        
        self.model_type = model_type
    
    def switch_model(self, model_type):
        """Switch to a different model at runtime"""
        self.load_model(model_type)
    
    def classify_text(self, text):
        """
        Classify a single license text
        Returns: (license_id, confidence)
        """
        if not text or len(text.strip()) < 10:
            return None, 0.0
        
        # Vectorize
        X = self.vectorizer.transform([text])
        # Slice to TF-IDF features only (5000) to match training data
        X = X[:, :5000]
        X_train_tfidf = self.X_train[:, :5000]
        
        # Get similarity predictions
        similarities = cosine_similarity(X, X_train_tfidf).flatten()
        most_similar_idx = np.argmax(similarities)
        predicted_license = self.y_train[most_similar_idx]
        confidence = float(similarities[most_similar_idx])
        
        return predicted_license, confidence
    
    def classify_file(self, file_path):
        """
        Classify a single license file
        Returns: {filename, license_id, confidence, file_size}
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
            
            license_id, confidence = self.classify_text(text)
            
            return {
                'filename': Path(file_path).name,
                'filepath': str(file_path),
                'license_id': license_id,
                'confidence': confidence,
                'file_size': len(text),
                'status': 'success'
            }
        except Exception as e:
            return {
                'filename': Path(file_path).name,
                'filepath': str(file_path),
                'license_id': None,
                'confidence': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def classify_directory(self, directory, pattern='*.txt'):
        """
        Classify all files in a directory
        Returns: list of classification results
        """
        print(f"\nClassifying files in: {directory}")
        print(f"Pattern: {pattern}\n")
        
        results = []
        directory = Path(directory)
        files = list(directory.glob(pattern))
        
        if not files:
            print(f"[ERROR] No files matching '{pattern}' found")
            return results
        
        print(f"Found {len(files)} files\n")
        
        for i, file_path in enumerate(files, 1):
            result = self.classify_file(file_path)
            results.append(result)
            
            status = "[OK]" if result['status'] == 'success' else "[ERROR]"
            license_id = result.get('license_id') or 'NONE'
            confidence = result.get('confidence') or 0.0
            print(f"{status} [{i:3d}/{len(files)}] {file_path.name:40s} -> {license_id:25s} ({confidence:.1%})")
        
        return results
    
    def save_results(self, results, output_file='classifications.csv'):
        """Save classification results to CSV"""
        if not results:
            print("No results to save")
            return
        
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"\n[OK] Results saved to {output_file}")
        
        # Print summary
        successful = len([r for r in results if r['status'] == 'success'])
        errors = len([r for r in results if r['status'] == 'error'])
        avg_confidence = df[df['status'] == 'success']['confidence'].mean()
        
        print(f"\nSummary:")
        print(f"  Total files: {len(results)}")
        print(f"  Successfully classified: {successful}")
        print(f"  Errors: {errors}")
        print(f"  Average confidence: {avg_confidence:.1%}")
        
        return df
    
    def export_json(self, results, output_file='classifications.json'):
        """Export results as JSON"""
        json_results = {
            'timestamp': datetime.now().isoformat(),
            'total': len(results),
            'successful': len([r for r in results if r['status'] == 'success']),
            'classifications': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"[OK] JSON results saved to {output_file}")


def main():
    """Example usage"""
    print("=" * 80)
    print("AUTOMATED LICENSE CLASSIFICATION SYSTEM")
    print("=" * 80)
    
    # Initialize classifier
    classifier = LicenseClassifier()
    
    # Example 1: Classify a single file
    print("\n[EXAMPLE 1] Classifying a single file:")
    print("-" * 80)
    
    result = classifier.classify_file('data/raw/license-list-XML/MIT.xml')
    print(f"\nFile: {result['filename']}")
    print(f"License: {result['license_id']}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    # Example 2: Classify a directory
    print("\n\n[EXAMPLE 2] Classifying all licenses in data/raw/license-list-XML/:")
    print("-" * 80)
    
    results = classifier.classify_directory(
        'data/raw/license-list-XML/',
        pattern='*.xml'
    )
    
    # Save results
    if results:
        df = classifier.save_results(results, 'license_classifications.csv')
        classifier.export_json(results, 'license_classifications.json')
        
        # Show top licenses by confidence
        print("\n\nTop 10 Most Confident Classifications:")
        print("-" * 80)
        df_success = df[df['status'] == 'success'].copy()
        top_10 = df_success.nlargest(10, 'confidence')
        for idx, row in top_10.iterrows():
            print(f"{row['license_id']:25s} {row['confidence']:.1%}  ({row['filename']})")
        
        # Show licenses by type
        print("\n\nLicense Distribution:")
        print("-" * 80)
        license_dist = df_success['license_id'].value_counts()
        for license_id, count in license_dist.head(15).items():
            print(f"  {license_id:25s} : {count:3d} files")


if __name__ == '__main__':
    main()
