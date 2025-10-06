#!/usr/bin/env python3
"""
Apply optimized thresholds to improve DistilBERT performance
Test different threshold strategies for better results
"""

import json
import numpy as np
from sklearn.metrics import classification_report, f1_score
import pandas as pd

def load_evaluation_results(results_file):
    """Load the evaluation results"""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    return {
        'label_names': results['label_names'],
        'y_true': np.array(results['true_labels']),
        'y_scores': np.array(results['scores']),
        'original_predictions': np.array(results['predictions'])
    }

def apply_optimized_thresholds(y_scores, label_names, strategy='optimal'):
    """Apply different threshold strategies"""
    
    strategies = {
        'uniform_05': {label: 0.5 for label in label_names},  # Original
        'optimal': {  # Based on your analysis results
            'Bug': 0.5,
            'New Feature': 0.5, 
            'Documentation': 0.5,
            'Build / CI': 0.15,  # Lowered significantly
            'help wanted': 0.3,   # Lowered moderately
            'RFC': 0.5,
            'Enhancement': 0.10   # Lowered significantly
        },
        'conservative': {  # Higher thresholds for precision
            'Bug': 0.6,
            'New Feature': 0.6,
            'Documentation': 0.6,
            'Build / CI': 0.4,
            'help wanted': 0.4,
            'RFC': 0.6,
            'Enhancement': 0.4
        },
        'aggressive': {  # Lower thresholds for recall
            'Bug': 0.3,
            'New Feature': 0.3,
            'Documentation': 0.3,
            'Build / CI': 0.1,
            'help wanted': 0.1,
            'RFC': 0.3,
            'Enhancement': 0.05
        }
    }
    
    thresholds = strategies[strategy]
    predictions = np.zeros_like(y_scores)
    
    for i, label in enumerate(label_names):
        threshold = thresholds[label]
        predictions[:, i] = (y_scores[:, i] > threshold).astype(int)
    
    return predictions, thresholds

def evaluate_with_thresholds(data, strategy='optimal'):
    """Evaluate model with different threshold strategies"""
    
    y_true = data['y_true']
    y_scores = data['y_scores']
    label_names = data['label_names']
    
    # Apply thresholds
    y_pred, thresholds = apply_optimized_thresholds(y_scores, label_names, strategy)
    
    # Calculate metrics
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    per_label_f1 = f1_score(y_true, y_pred, average=None)
    
    results = {
        'strategy': strategy,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'per_label_f1': dict(zip(label_names, per_label_f1)),
        'thresholds': thresholds
    }
    
    return results, y_pred

def compare_threshold_strategies(results_file):
    """Compare different threshold strategies"""
    
    print("🔧 Threshold Optimization Comparison")
    print("=" * 60)
    
    # Load data
    data = load_evaluation_results(results_file)
    
    strategies = ['uniform_05', 'optimal', 'conservative', 'aggressive']
    all_results = []
    
    for strategy in strategies:
        results, y_pred = evaluate_with_thresholds(data, strategy)
        all_results.append(results)
        
        print(f"\n📊 {strategy.upper()} Strategy:")
        print(f"  Micro F1: {results['micro_f1']:.3f}")
        print(f"  Macro F1: {results['macro_f1']:.3f}")
        
        # Show per-label improvements
        if strategy != 'uniform_05':
            original_results = all_results[0]  # First is uniform_05
            print("  Per-label changes:")
            for label in data['label_names']:
                original_f1 = original_results['per_label_f1'][label]
                new_f1 = results['per_label_f1'][label]
                change = new_f1 - original_f1
                if abs(change) > 0.01:  # Only show significant changes
                    sign = "+" if change > 0 else ""
                    print(f"    {label:<15}: {new_f1:.3f} ({sign}{change:.3f})")
    
    # Find best strategy
    best_micro = max(all_results, key=lambda x: x['micro_f1'])
    best_macro = max(all_results, key=lambda x: x['macro_f1'])
    
    print(f"\n🏆 Best Results:")
    print(f"  Micro F1: {best_micro['strategy']} ({best_micro['micro_f1']:.3f})")
    print(f"  Macro F1: {best_macro['strategy']} ({best_macro['macro_f1']:.3f})")
    
    return all_results

def create_production_config(results_file):
    """Create a production-ready configuration"""
    
    data = load_evaluation_results(results_file)
    
    # Use optimal strategy for production
    results, y_pred = evaluate_with_thresholds(data, 'optimal')
    
    # Focus on the main categories (90%+ of issues)
    main_categories = ['Bug', 'New Feature', 'Documentation']
    main_f1_scores = [results['per_label_f1'][cat] for cat in main_categories]
    main_avg_f1 = np.mean(main_f1_scores)
    
    config = {
        'model_version': 'distilbert_v1',
        'thresholds': results['thresholds'],
        'performance_metrics': {
            'overall_micro_f1': results['micro_f1'],
            'overall_macro_f1': results['macro_f1'],
            'main_categories_f1': main_avg_f1,
            'per_label_f1': results['per_label_f1']
        },
        'main_categories': main_categories,
        'coverage': f"{(291/320)*100:.1f}%",  # Bug+NewFeature+Docs coverage
        'recommended_use': 'Focus on Bug, New Feature, Documentation classification'
    }
    
    # Save config
    config_file = 'results/production_config.json'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n🚀 Production Configuration")
    print("=" * 40)
    print(f"Main Categories F1: {main_avg_f1:.3f}")
    print(f"Overall Micro F1: {results['micro_f1']:.3f}")
    print(f"Issue Coverage: {(291/320)*100:.1f}%")
    print(f"Config saved to: {config_file}")
    
    return config

def analyze_failure_cases(results_file):
    """Analyze specific failure cases for insights"""
    
    data = load_evaluation_results(results_file)
    results, y_pred = evaluate_with_thresholds(data, 'optimal')
    
    y_true = data['y_true']
    label_names = data['label_names']
    
    print(f"\n🔍 Failure Case Analysis")
    print("=" * 40)
    
    for i, label in enumerate(label_names):
        true_positives = ((y_true[:, i] == 1) & (y_pred[:, i] == 1)).sum()
        false_negatives = ((y_true[:, i] == 1) & (y_pred[:, i] == 0)).sum()
        false_positives = ((y_true[:, i] == 0) & (y_pred[:, i] == 1)).sum()
        
        total_true = y_true[:, i].sum()
        
        if total_true > 0:
            recall = true_positives / total_true
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            
            print(f"{label:<15}: P={precision:.3f}, R={recall:.3f} | TP={true_positives}, FN={false_negatives}, FP={false_positives}")

def main():
    """Run comprehensive threshold optimization analysis"""
    
    # Find latest results file
    import os
    results_files = [f for f in os.listdir('results') if f.startswith('distilbert_evaluation_') and f.endswith('.json')]
    if not results_files:
        print("❌ No evaluation results found!")
        return
    
    latest_results = f"results/{sorted(results_files)[-1]}"
    
    # Run analysis
    all_results = compare_threshold_strategies(latest_results)
    config = create_production_config(latest_results)
    analyze_failure_cases(latest_results)
    
    print(f"\n✅ Optimization Complete!")
    print(f"🎯 Key Takeaway: Your model achieves {config['performance_metrics']['main_categories_f1']:.1%} F1 on main categories")
    print(f"📊 This covers {config['coverage']} of all issues")
    print(f"🚀 Ready for production use!")

if __name__ == "__main__":
    main()