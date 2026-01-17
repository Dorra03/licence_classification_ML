"""
Production ML Predictor for License Classification
Loads trained models and provides prediction interface
"""

import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

class LicenseClassifier:
    """Production-ready license classifier"""
    
    def __init__(self, model_type='gradient_boosting'):
        """
        Initialize classifier with trained model
        
        Args:
            model_type: 'gradient_boosting', 'random_forest', or 'naive_bayes'
        """
        self.model_type = model_type
        self.model_dir = Path("models")
        
        self._load_models()
        self._load_vectorizers()
    
    def _load_models(self):
        """Load trained models"""
        model_files = {
            'gradient_boosting': self.model_dir / 'gradient_boosting_model.pkl',
            'random_forest': self.model_dir / 'random_forest_model.pkl',
            'naive_bayes': self.model_dir / 'naive_bayes_model.pkl',
        }
        
        if self.model_type not in model_files:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model_path = model_files[self.model_type]
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"[OK] Loaded {self.model_type} model")
    
    def _load_vectorizers(self):
        """Load TF-IDF vectorizer and label encoder"""
        tfidf_path = self.model_dir / 'tfidf_vectorizer.pkl'
        label_encoder_path = self.model_dir / 'label_encoder_license.pkl'
        
        if not tfidf_path.exists():
            raise FileNotFoundError(f"TF-IDF vectorizer not found: {tfidf_path}")
        if not label_encoder_path.exists():
            raise FileNotFoundError(f"Label encoder not found: {label_encoder_path}")
        
        with open(tfidf_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        with open(label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        print(f"[OK] Loaded feature vectorizer and label encoder")
    
    def predict(self, text):
        """
        Predict license category from text
        
        Args:
            text (str): License text
        
        Returns:
            dict: Prediction result with category and confidence
        """
        # Vectorize text
        X = self.vectorizer.transform([text])
        
        # Predict
        pred = self.model.predict(X)[0]
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0
        
        # Decode label
        category = self.label_encoder.inverse_transform([pred])[0]
        
        return {
            'category': category,
            'confidence': confidence,
            'model': self.model_type
        }
    
    def predict_batch(self, texts):
        """
        Predict categories for multiple texts
        
        Args:
            texts (list): List of license texts
        
        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            results.append(self.predict(text))
        return results

def demo():
    """Demo prediction"""
    import json
    
    # Initialize classifier
    classifier = LicenseClassifier(model_type='gradient_boosting')
    
    # Example texts
    examples = {
        'MIT': 'Permission is hereby granted, free of charge...',
        'GPL': 'This program is free software; you can redistribute it',
        'Apache': 'Licensed under the Apache License, Version 2.0',
    }
    
    print("\n" + "="*60)
    print("DEMO: LICENSE CLASSIFICATION")
    print("="*60)
    
    for name, text in examples.items():
        result = classifier.predict(text)
        print(f"\nInput: {name}")
        print(f"Prediction: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    demo()
