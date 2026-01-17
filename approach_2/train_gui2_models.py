"""
Train multiple classification models for license classification.
Follows the same approach as approach_2 - trains and saves models.
"""

import pickle
import json
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for XGBoost and LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available")


class MultiModelTrainer:
    """Train and evaluate multiple classification models."""
    
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.results = {}
        
    def load_training_data(self):
        """Load training data from approach_2 training directory."""
        logger.info("Loading training data...")
        
        try:
            # Load from approach_2 data
            X = pd.read_csv('data/processed/X_train_processed.csv')
            y = pd.read_csv('data/processed/y_train_processed.csv')
            
            texts = X.iloc[:, 0].values
            labels = y.iloc[:, 0].values
            
            logger.info(f"Loaded {len(texts)} training samples")
            return texts, labels
            
        except Exception as e:
            logger.warning(f"Could not load processed data: {e}")
            logger.info("Using sample data instead...")
            return self._get_sample_data()
    
    def _get_sample_data(self):
        """Generate sample training data if processed data not available."""
        sample_texts = [
            "MIT License Permission is hereby granted",
            "GNU General Public License version 2",
            "Apache License 2.0 Licensed under",
            "BSD License Redistribution and use",
            "GPL v3 or later version",
            "ISC License Permission to use",
            "Mozilla Public License Version 2.0",
            "LGPL License Library or Lesser",
            "Unlicense This software is released",
            "WTFPL Do what the F you want",
            "Creative Commons Attribution 4.0",
            "Boost Software License 1.0",
        ]
        
        sample_labels = [
            'permissive',
            'copyleft',
            'permissive',
            'permissive',
            'copyleft',
            'permissive',
            'permissive',
            'permissive',
            'other',
            'other',
            'other',
            'permissive',
        ]
        
        # Repeat samples to create larger dataset
        texts = sample_texts * 50
        labels = sample_labels * 50
        
        logger.info(f"Generated {len(texts)} sample training samples")
        return texts, labels
    
    def prepare_features(self, texts):
        """Vectorize text using TF-IDF."""
        logger.info("Preparing TF-IDF features...")
        
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            lowercase=True,
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        
        X = self.vectorizer.fit_transform(texts)
        logger.info(f"Feature matrix shape: {X.shape}")
        return X
    
    def train_models(self, X, y):
        """Train multiple classification models."""
        logger.info("\n" + "="*70)
        logger.info("STARTING MODEL TRAINING")
        logger.info("="*70)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Train size: {self.X_train.shape[0]}, Test size: {self.X_test.shape[0]}")
        
        # 1. Gradient Boosting (Primary Model)
        logger.info("\n[1/7] Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        gb_model.fit(self.X_train, self.y_train)
        self.models['gradient_boosting'] = gb_model
        self._evaluate_model('gradient_boosting', gb_model)
        
        # 2. Random Forest
        logger.info("\n[2/7] Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(self.X_train, self.y_train)
        self.models['random_forest'] = rf_model
        self._evaluate_model('random_forest', rf_model)
        
        # 3. Naive Bayes
        logger.info("\n[3/7] Training Naive Bayes...")
        nb_model = MultinomialNB()
        nb_model.fit(self.X_train, self.y_train)
        self.models['naive_bayes'] = nb_model
        self._evaluate_model('naive_bayes', nb_model)
        
        # 4. SVM
        logger.info("\n[4/7] Training Linear SVM...")
        svm_model = LinearSVC(
            max_iter=2000,
            random_state=42,
            dual=False
        )
        svm_model.fit(self.X_train, self.y_train)
        self.models['svm'] = svm_model
        self._evaluate_model('svm', svm_model)
        
        # 5. Neural Network (MLP)
        logger.info("\n[5/7] Training Neural Network (MLP)...")
        try:
            nn_model = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=500,
                random_state=42,
                early_stopping=False,
                solver='adam'
            )
            nn_model.fit(self.X_train, self.y_train)
            self.models['neural_network'] = nn_model
            self._evaluate_model('neural_network', nn_model)
        except Exception as e:
            logger.error(f"Neural Network training failed: {e}")
            logger.info("  Skipping Neural Network model")
        
        # 6. XGBoost
        if XGBOOST_AVAILABLE:
            logger.info("\n[6/7] Training XGBoost...")
            try:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    eval_metric='mlogloss'
                )
                xgb_model.fit(self.X_train, self.y_train)
                self.models['xgboost'] = xgb_model
                self._evaluate_model('xgboost', xgb_model)
            except Exception as e:
                logger.error(f"XGBoost training failed: {e}")
        else:
            logger.info("\n[6/7] XGBoost skipped (not installed)")
        
        # 7. LightGBM
        if LIGHTGBM_AVAILABLE:
            logger.info("\n[7/7] Training LightGBM...")
            try:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=42,
                    verbose=-1
                )
                lgb_model.fit(self.X_train, self.y_train)
                self.models['lightgbm'] = lgb_model
                self._evaluate_model('lightgbm', lgb_model)
            except Exception as e:
                logger.error(f"LightGBM training failed: {e}")
        else:
            logger.info("\n[7/7] LightGBM skipped (not installed)")
        
        logger.info("\n" + "="*70)
        logger.info("ALL MODELS TRAINED SUCCESSFULLY")
        logger.info("="*70)
    
    def _evaluate_model(self, model_name, model):
        """Evaluate a single model."""
        y_pred = model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        self.results[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
    
    def save_models(self, output_dir='models'):
        """Save trained models and vectorizer."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"\nSaving models to {output_dir}...")
        
        # Save vectorizer
        with open(f'{output_dir}/vectorizer.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logger.info(f"  ✓ vectorizer.pkl saved")
        
        # Save models
        for name, model in self.models.items():
            model_path = f'{output_dir}/{name}_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"  ✓ {name}_model.pkl saved")
        
        # Save results
        results_path = f'{output_dir}/training_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"  ✓ training_results.json saved")
        
        logger.info("All models saved successfully!")
    
    def print_summary(self):
        """Print training summary."""
        logger.info("\n" + "="*70)
        logger.info("TRAINING SUMMARY")
        logger.info("="*70)
        
        # Sort by accuracy
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        logger.info("\nModel Rankings (by Accuracy):\n")
        for rank, (name, metrics) in enumerate(sorted_results, 1):
            logger.info(
                f"[{rank}] {name.upper():20s} - "
                f"Accuracy: {metrics['accuracy']*100:6.2f}% | "
                f"F1: {metrics['f1']:.4f}"
            )
        
        # Best model
        best_model = sorted_results[0]
        logger.info(f"\nBest Model: {best_model[0].upper()}")
        logger.info(f"Accuracy: {best_model[1]['accuracy']*100:.2f}%")
        
        logger.info("\n" + "="*70)


def main():
    """Main training pipeline."""
    trainer = MultiModelTrainer()
    
    # Load data
    texts, labels = trainer.load_training_data()
    
    # Prepare features
    X = trainer.prepare_features(texts)
    
    # Train models
    trainer.train_models(X, labels)
    
    # Save models
    trainer.save_models()
    
    # Print summary
    trainer.print_summary()


if __name__ == '__main__':
    main()
