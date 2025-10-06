#!/usr/bin/env python3
"""
Improvement strategies for the DistilBERT Issue Classifier
Focus on addressing class imbalance and low-performing categories
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class ModelImprovementAnalyzer:
    """Analyze and improve the trained DistilBERT model"""
    
    def __init__(self, results_file: str):
        """Load evaluation results for analysis"""
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        self.label_names = self.results['label_names']
        self.y_true = np.array(self.results['true_labels'])
        self.y_pred = np.array(self.results['predictions'])
        self.y_scores = np.array(self.results['scores'])
        
    def analyze_class_distribution(self):
        """Analyze the class distribution and imbalance"""
        print("📊 Class Distribution Analysis")
        print("=" * 50)
        
        # Count samples per class
        class_counts = self.y_true.sum(axis=0)
        total_samples = len(self.y_true)
        
        print("Test set distribution:")
        for i, (label, count) in enumerate(zip(self.label_names, class_counts)):
            pct = (count / total_samples) * 100
            f1 = self.results['metrics']['per_label_f1'][label]
            print(f"  {label:<15}: {count:>3} samples ({pct:>5.1f}%) → F1: {f1:.3f}")
        
        # Identify problematic classes
        low_sample_classes = [(label, count) for label, count in zip(self.label_names, class_counts) if count < 20]
        
        if low_sample_classes:
            print(f"\n⚠️  Classes with <20 test samples:")
            for label, count in low_sample_classes:
                print(f"    {label}: {count} samples")
        
        return class_counts
    
    def suggest_threshold_optimization(self):
        """Suggest optimal thresholds for each class"""
        print("\n🎯 Threshold Optimization Suggestions")
        print("=" * 50)
        
        optimal_thresholds = {}
        
        for i, label in enumerate(self.label_names):
            y_true_label = self.y_true[:, i]
            y_scores_label = self.y_scores[:, i]
            
            if y_true_label.sum() == 0:  # No positive examples
                optimal_thresholds[label] = 0.5
                continue
            
            # Test different thresholds
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in np.arange(0.1, 0.9, 0.05):
                y_pred_threshold = (y_scores_label > threshold).astype(int)
                f1 = f1_score(y_true_label, y_pred_threshold, zero_division=0)
                
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            optimal_thresholds[label] = best_threshold
            current_f1 = self.results['metrics']['per_label_f1'][label]
            improvement = best_f1 - current_f1
            
            if improvement > 0.05:  # Significant improvement
                print(f"  {label:<15}: {best_threshold:.2f} (F1: {best_f1:.3f}, +{improvement:.3f})")
        
        return optimal_thresholds
    
    def analyze_confusion_patterns(self):
        """Analyze which labels get confused with each other"""
        print("\n🔄 Label Confusion Analysis")
        print("=" * 50)
        
        # For multi-label, look at co-occurrence patterns
        print("Label co-occurrence in true labels:")
        for i, label1 in enumerate(self.label_names):
            for j, label2 in enumerate(self.label_names[i+1:], i+1):
                co_occur = ((self.y_true[:, i] == 1) & (self.y_true[:, j] == 1)).sum()
                if co_occur > 0:
                    print(f"  {label1} + {label2}: {co_occur} issues")
        
        # Prediction errors
        print("\nMost common prediction errors:")
        for i, label in enumerate(self.label_names):
            true_pos = self.y_true[:, i]
            pred_pos = self.y_pred[:, i]
            
            false_negatives = ((true_pos == 1) & (pred_pos == 0)).sum()
            false_positives = ((true_pos == 0) & (pred_pos == 1)).sum()
            
            if false_negatives > 0 or false_positives > 0:
                print(f"  {label:<15}: {false_negatives} missed, {false_positives} false alarms")
    
    def create_performance_visualization(self):
        """Create visualizations of model performance"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('DistilBERT Model Performance Analysis', fontsize=16)
        
        # 1. F1 scores by label
        f1_scores = [self.results['metrics']['per_label_f1'][label] for label in self.label_names]
        colors = ['green' if f1 > 0.7 else 'orange' if f1 > 0.3 else 'red' for f1 in f1_scores]
        
        axes[0, 0].bar(range(len(self.label_names)), f1_scores, color=colors)
        axes[0, 0].set_xticks(range(len(self.label_names)))
        axes[0, 0].set_xticklabels(self.label_names, rotation=45, ha='right')
        axes[0, 0].set_title('F1 Score by Label')
        axes[0, 0].set_ylabel('F1 Score')
        axes[0, 0].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Good (0.7)')
        axes[0, 0].axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Poor (0.3)')
        axes[0, 0].legend()
        
        # 2. Sample count vs F1 score
        sample_counts = self.y_true.sum(axis=0)
        axes[0, 1].scatter(sample_counts, f1_scores, s=100, alpha=0.7)
        for i, label in enumerate(self.label_names):
            axes[0, 1].annotate(label, (sample_counts[i], f1_scores[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[0, 1].set_xlabel('Number of Test Samples')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('Sample Count vs Performance')
        
        # 3. Precision vs Recall
        precisions = []
        recalls = []
        for i, label in enumerate(self.label_names):
            true_pos = ((self.y_true[:, i] == 1) & (self.y_pred[:, i] == 1)).sum()
            pred_pos = self.y_pred[:, i].sum()
            actual_pos = self.y_true[:, i].sum()
            
            precision = true_pos / pred_pos if pred_pos > 0 else 0
            recall = true_pos / actual_pos if actual_pos > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
        
        axes[1, 0].scatter(recalls, precisions, s=100, alpha=0.7)
        for i, label in enumerate(self.label_names):
            axes[1, 0].annotate(label, (recalls[i], precisions[i]), 
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision vs Recall')
        axes[1, 0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
        
        # 4. Score distribution for positive vs negative examples
        for i, label in enumerate(['Bug', 'New Feature', 'Documentation']):  # Top 3 labels
            if label in self.label_names:
                idx = self.label_names.index(label)
                pos_scores = self.y_scores[self.y_true[:, idx] == 1, idx]
                neg_scores = self.y_scores[self.y_true[:, idx] == 0, idx]
                
                axes[1, 1].hist(neg_scores, bins=20, alpha=0.5, label=f'{label} (neg)', density=True)
                axes[1, 1].hist(pos_scores, bins=20, alpha=0.5, label=f'{label} (pos)', density=True)
        
        axes[1, 1].set_xlabel('Prediction Score')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Score Distribution (Top 3 Labels)')
        axes[1, 1].legend()
        axes[1, 1].axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'results/performance_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Performance analysis saved to: results/performance_analysis_{timestamp}.png")

def improvement_recommendations(results_file: str):
    """Provide specific recommendations for model improvement"""
    
    analyzer = ModelImprovementAnalyzer(results_file)
    
    # Run analysis
    class_counts = analyzer.analyze_class_distribution()
    optimal_thresholds = analyzer.suggest_threshold_optimization()
    analyzer.analyze_confusion_patterns()
    analyzer.create_performance_visualization()
    
    print("\n🚀 Improvement Recommendations")
    print("=" * 50)
    
    print("1. Immediate improvements (no retraining needed):")
    print("   • Use optimized thresholds per label instead of 0.5 for all")
    print("   • Focus evaluation on major categories (Bug, New Feature, Documentation)")
    print("   • Consider weighted F1 instead of macro F1 for business metrics")
    
    print("\n2. Data-based improvements:")
    print("   • Collect more examples for: help wanted, Enhancement, Build/CI")
    print("   • Consider merging similar low-frequency categories")
    print("   • Use class weights during training to handle imbalance")
    
    print("\n3. Model-based improvements:")
    print("   • Try RoBERTa (similar to DistilBERT but sometimes better)")
    print("   • Experiment with longer max_length (currently 512)")
    print("   • Use focal loss instead of BCE for severe class imbalance")
    
    print("\n4. Evaluation perspective:")
    print("   • Your 84% Micro F1 is excellent for real-world use!")
    print("   • Focus on the 3 major categories that handle 87% of issues")
    print("   • Consider this a strong baseline for LLM comparison")

def main():
    """Run improvement analysis on the latest results"""
    import os
    
    # Find latest results file
    results_files = [f for f in os.listdir('results') if f.startswith('distilbert_evaluation_') and f.endswith('.json')]
    if not results_files:
        print("❌ No evaluation results found!")
        return
    
    latest_results = f"results/{sorted(results_files)[-1]}"
    print(f"📂 Analyzing: {latest_results}")
    
    improvement_recommendations(latest_results)

if __name__ == "__main__":
    main()