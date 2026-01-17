"""
Phase 4: Evaluation and Visualization
- Evaluate all models using cosine similarity (like Approach 1)
- Generate visualizations
- Create comprehensive analysis report
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def load_features():
    """Load features"""
    logger.info("Loading features...")
    
    X_train = np.load(PROCESSED_DIR / "X_train.npz")['X_train']
    X_test = np.load(PROCESSED_DIR / "X_test.npz")['X_test']
    y_train = np.load(PROCESSED_DIR / "y_train.npz")['y_train']
    y_test = np.load(PROCESSED_DIR / "y_test.npz")['y_test']
    
    return X_train, X_test, y_train, y_test

def load_models():
    """Load all trained models"""
    logger.info("Loading trained models...")
    
    models = {}
    
    for model_file in MODEL_DIR.glob("*_model.pkl"):
        model_name = model_file.stem.replace("_model", "")
        with open(model_file, "rb") as f:
            models[model_name] = pickle.load(f)
        logger.info(f"  Loaded: {model_name}")
    
    return models

def evaluate_with_similarity(X_train, X_test, y_train, y_test, model, model_name):
    """Evaluate model using cosine similarity (like Approach 1)"""
    logger.info(f"\nEvaluating {model_name} with cosine similarity...")
    
    # Get prediction probabilities/distances
    if hasattr(model, 'predict_proba'):
        y_train_proba = model.predict_proba(X_train)
        y_test_proba = model.predict_proba(X_test)
    else:
        # For models without proba, use decision function or raw predictions
        y_train_proba = normalize(model.predict(X_train).reshape(-1, 1), norm='l2')
        y_test_proba = normalize(model.predict(X_test).reshape(-1, 1), norm='l2')
    
    # Compute cosine similarity between test and training predictions
    # This measures how similar test predictions are to training space
    similarities = []
    for test_pred in y_test_proba:
        # Find max similarity to any training prediction
        max_sim = np.max(cosine_similarity([test_pred], y_train_proba))
        similarities.append(max_sim)
    
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    
    # Standard accuracy
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    metrics = {
        'model': model_name,
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'avg_similarity': float(avg_similarity),
        'std_similarity': float(std_similarity),
        'unique_train_predictions': int(len(np.unique(y_train_pred))),
        'unique_test_predictions': int(len(np.unique(y_test_pred)))
    }
    
    logger.info(f"  Train Accuracy: {train_acc:.4f}")
    logger.info(f"  Test Accuracy: {test_acc:.4f}")
    logger.info(f"  Avg Similarity: {avg_similarity:.4f} ± {std_similarity:.4f}")
    
    return metrics, y_test_pred, similarities

def create_comparison_report(all_metrics):
    """Create comparison report"""
    logger.info("\nCreating comparison report...")
    
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'models_evaluated': len(all_metrics),
        'metrics': all_metrics,
        'best_model': max(all_metrics, key=lambda x: x['test_accuracy']),
        'best_similarity': max(all_metrics, key=lambda x: x['avg_similarity']),
    }
    
    # Save report
    with open(RESULTS_DIR / "evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"✓ Saved to {RESULTS_DIR / 'evaluation_report.json'}")
    
    return report

def create_visualizations(all_metrics):
    """Create comparison visualizations"""
    logger.info("\nCreating visualizations...")
    
    df_metrics = pd.DataFrame(all_metrics)
    
    # 1. Model Comparison - Accuracy
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Accuracy comparison
    ax = axes[0, 0]
    x = np.arange(len(df_metrics))
    width = 0.35
    ax.bar(x - width/2, df_metrics['train_accuracy'], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, df_metrics['test_accuracy'], width, label='Test', alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Similarity comparison
    ax = axes[0, 1]
    ax.errorbar(df_metrics['model'], df_metrics['avg_similarity'], 
                yerr=df_metrics['std_similarity'], 
                fmt='o-', capsize=5, capthick=2)
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Model Similarity Metrics')
    ax.set_xticklabels(df_metrics['model'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Unique predictions
    ax = axes[1, 0]
    x = np.arange(len(df_metrics))
    ax.bar(x - width/2, df_metrics['unique_train_predictions'], width, label='Train', alpha=0.8)
    ax.bar(x + width/2, df_metrics['unique_test_predictions'], width, label='Test', alpha=0.8)
    ax.set_xlabel('Model')
    ax.set_ylabel('Unique Predictions')
    ax.set_title('Prediction Diversity')
    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Best metrics summary
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = "BEST MODELS\n" + "="*40 + "\n"
    
    best_acc = df_metrics.loc[df_metrics['test_accuracy'].idxmax()]
    best_sim = df_metrics.loc[df_metrics['avg_similarity'].idxmax()]
    
    summary_text += f"\nBy Accuracy:\n{best_acc['model']}\nTest Acc: {best_acc['test_accuracy']:.4f}\n"
    summary_text += f"\nBy Similarity:\n{best_sim['model']}\nAvg Sim: {best_sim['avg_similarity']:.4f}\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=300, bbox_inches='tight')
    logger.info(f"  Saved: model_comparison.png")
    plt.close()
    
    # 2. Detailed metrics table
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for _, row in df_metrics.iterrows():
        table_data.append([
            row['model'],
            f"{row['train_accuracy']:.4f}",
            f"{row['test_accuracy']:.4f}",
            f"{row['avg_similarity']:.4f} ± {row['std_similarity']:.4f}",
            f"{int(row['unique_test_predictions'])}"
        ])
    
    table = ax.table(cellText=table_data,
                     colLabels=['Model', 'Train Acc', 'Test Acc', 'Similarity', 'Unique Pred'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.title('Approach 2 - Model Evaluation Metrics', fontsize=14, pad=20)
    plt.savefig(RESULTS_DIR / "metrics_table.png", dpi=300, bbox_inches='tight')
    logger.info(f"  Saved: metrics_table.png")
    plt.close()

def create_final_report():
    """Create comprehensive final report"""
    logger.info("\nCreating final comprehensive report...")
    
    with open(RESULTS_DIR / "training_metrics.json") as f:
        training_metrics = json.load(f)
    
    with open(RESULTS_DIR / "evaluation_report.json") as f:
        eval_report = json.load(f)
    
    report_text = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                     APPROACH 2 - FINAL EVALUATION REPORT                   ║
║                                                                            ║
║                          PHASE 1-4 COMPLETE ✅                           ║
╚════════════════════════════════════════════════════════════════════════════╝

DATASET SUMMARY
───────────────────────────────────────────────────────────────────────────
  Total Samples:           {eval_report.get('metrics', [{}])[0].get('total_samples', 'N/A')}
  Training Samples:        {eval_report.get('metrics', [{}])[0].get('train_samples', 'N/A')}
  Test Samples:            {eval_report.get('metrics', [{}])[0].get('test_samples', 'N/A')}
  Feature Dimensions:      {eval_report.get('metrics', [{}])[0].get('feature_dimensions', 'N/A')}
  License Classes:         {eval_report.get('metrics', [{}])[0].get('n_classes', 'N/A')}

TRAINING METRICS (Best Models)
───────────────────────────────────────────────────────────────────────────
"""
    
    for i, metrics in enumerate(training_metrics[:3], 1):
        report_text += f"\n{i}. {metrics['model']}\n"
        report_text += f"   Train Accuracy: {metrics['train_accuracy']:.4f}\n"
        report_text += f"   Test Accuracy:  {metrics['test_accuracy']:.4f}\n"
        report_text += f"   Training Time:  {metrics['training_time']:.2f}s\n"
    
    report_text += f"""

EVALUATION METRICS
───────────────────────────────────────────────────────────────────────────
  Best Model (Accuracy): {eval_report['best_model']['model']}
    - Test Accuracy: {eval_report['best_model']['test_accuracy']:.4f}
    - Similarity: {eval_report['best_model']['avg_similarity']:.4f}

  Best Model (Similarity): {eval_report['best_similarity']['model']}
    - Test Accuracy: {eval_report['best_similarity']['test_accuracy']:.4f}
    - Similarity: {eval_report['best_similarity']['avg_similarity']:.4f}

COMPARISON WITH APPROACH 1
───────────────────────────────────────────────────────────────────────────
  Approach 1:
    - Training Samples: 574
    - Test Accuracy: 91.8%
    - Similarity: 0.8721

  Approach 2:
    - Training Samples: {eval_report.get('metrics', [{}])[0].get('train_samples', 'N/A')}
    - Best Test Accuracy: {eval_report['best_model']['test_accuracy']:.4f}
    - Best Similarity: {eval_report['best_similarity']['avg_similarity']:.4f}

✅ APPROACH 2 EXECUTION COMPLETE
───────────────────────────────────────────────────────────────────────────
  Phase 1: Data Preparation ✓
  Phase 2: Data Processing ✓
  Phase 3: Feature Engineering & Training ✓
  Phase 4: Evaluation & Visualization ✓

All models saved to: {MODEL_DIR}
Results saved to: {RESULTS_DIR}
Visualizations saved to: {RESULTS_DIR}

═══════════════════════════════════════════════════════════════════════════════
Generated: {pd.Timestamp.now().isoformat()}
═══════════════════════════════════════════════════════════════════════════════
"""
    
    # Save report
    report_file = RESULTS_DIR / "APPROACH2_FINAL_REPORT.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info(f"✓ Saved comprehensive report to {report_file}")
    
    # Print to console
    logger.info(report_text)

def main():
    """Main execution"""
    logger.info("=" * 80)
    logger.info("PHASE 4: EVALUATION & VISUALIZATION")
    logger.info("=" * 80)
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_features()
        models = load_models()
        
        if not models:
            logger.error("No models found!")
            return False
        
        # Evaluate all models
        all_metrics = []
        for model_name, model in models.items():
            metrics, _, _ = evaluate_with_similarity(X_train, X_test, y_train, y_test, model, model_name)
            all_metrics.append(metrics)
        
        # Create comparison report
        report = create_comparison_report(all_metrics)
        
        # Create visualizations
        create_visualizations(all_metrics)
        
        # Create final report
        create_final_report()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ PHASE 4: EVALUATION COMPLETE")
        logger.info(f"  Models Evaluated: {len(all_metrics)}")
        logger.info(f"  Results Saved: {RESULTS_DIR}")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in evaluation: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
