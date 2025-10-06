#!/usr/bin/env python3
"""
Complete DeBERTa Multi-label Issue Classifier (Full Training)
Same comprehensive treatment as DistilBERT and RoBERTa
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DebertaTokenizer, 
    DebertaForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import os
import logging
from typing import List, Dict, Tuple
import time
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device detection
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("🍎 Using Apple Silicon MPS acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("🚀 Using CUDA GPU")
else:
    device = torch.device("cpu")
    logger.info("💻 Using CPU")

class DebertaIssueDataset(Dataset):
    """Dataset class for DeBERTa"""
    
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Same label setup as other models
        self.label_names = ['Bug', 'New Feature', 'Documentation', 'Build / CI', 
                           'help wanted', 'RFC', 'Enhancement']
        self.mlb = MultiLabelBinarizer(classes=self.label_names)
        self.binary_labels = self.mlb.fit_transform(labels)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.binary_labels[idx].astype(np.float32)
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

class DebertaMultiLabelClassifier(nn.Module):
    """DeBERTa model for multi-label classification"""
    
    def __init__(self, num_labels: int, dropout_rate: float = 0.3):
        super().__init__()
        self.num_labels = num_labels
        
        # Load pre-trained DeBERTa
        self.deberta = DebertaForSequenceClassification.from_pretrained(
            'microsoft/deberta-base',
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.deberta.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.deberta.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {'loss': loss, 'logits': logits}

class DebertaTrainer:
    """Complete DeBERTa trainer (same comprehensive approach)"""
    
    def __init__(self):
        self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        self.label_names = ['Bug', 'New Feature', 'Documentation', 'Build / CI', 
                           'help wanted', 'RFC', 'Enhancement']
        self.num_labels = len(self.label_names)
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data"""
        data_files = [f for f in os.listdir('data') if f.startswith('sklearn_train_') and f.endswith('.json')]
        if not data_files:
            raise FileNotFoundError("No training data found!")
        
        train_file = f"data/{sorted(data_files)[-1]}"
        test_file = train_file.replace('train_', 'test_')
        
        logger.info(f"Loading training data from: {train_file}")
        logger.info(f"Loading test data from: {test_file}")
        
        train_df = pd.read_json(train_file)
        test_df = pd.read_json(test_file)
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        
        return train_df, test_df
    
    def prepare_datasets(self, train_df: pd.DataFrame, test_df: pd.DataFrame, max_length: int = 512):
        """Prepare datasets for training"""
        self.train_dataset = DebertaIssueDataset(
            texts=train_df['text'].tolist(),
            labels=train_df['labels'].tolist(),
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        self.val_dataset = DebertaIssueDataset(
            texts=test_df['text'].tolist(),
            labels=test_df['labels'].tolist(),
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        logger.info(f"Created datasets - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}")
        
        # Print label distribution
        train_labels = self.train_dataset.binary_labels
        label_counts = train_labels.sum(axis=0)
        
        print("\n📊 Training Label Distribution:")
        for i, (label, count) in enumerate(zip(self.label_names, label_counts)):
            pct = (count / len(train_labels)) * 100
            print(f"  {label:<15}: {count:>4} ({pct:>5.1f}%)")
    
    def train_model(self, epochs: int = 3, batch_size: int = 4, learning_rate: float = 2e-5):
        """Complete training pipeline (same as RoBERTa)"""
        
        logger.info("🚀 Starting DeBERTa training...")
        
        # Load data
        train_df, test_df = self.load_data()
        self.prepare_datasets(train_df, test_df)
        
        # Create data loaders
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0)
        
        # Initialize model
        self.model = DebertaMultiLabelClassifier(self.num_labels)
        self.model.to(device)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=total_steps
        )
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            logger.info(f"📚 Epoch {epoch + 1}/{epochs}")
            
            # Training phase
            self.model.train()
            total_train_loss = 0
            
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
            for batch in train_pbar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                total_train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            total_val_loss = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
                for batch in val_pbar:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = self.model(input_ids, attention_mask, labels)
                    loss = outputs['loss']
                    total_val_loss += loss.item()
                    val_pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
            
            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            logger.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Save model
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_path = f"models/deberta_classifier_{timestamp}"
        os.makedirs("models", exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_names': self.label_names,
            'tokenizer': self.tokenizer,
            'training_args': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'max_length': 512
            }
        }, f"{model_path}.pth")
        
        logger.info(f"💾 Model saved to: {model_path}.pth")
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, timestamp)
        
        return train_losses, val_losses, f"{model_path}.pth"
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float], timestamp: str):
        """Plot training curves (same as other models)"""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.title('DeBERTa Training Progress', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/deberta_training_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"📈 Training curves saved to: results/deberta_training_curves_{timestamp}.png")
    
    def evaluate_model(self, threshold: float = 0.5) -> Dict:
        """Complete evaluation (same as other models)"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        logger.info("📊 Evaluating model...")
        
        self.model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                outputs = self.model(input_ids, attention_mask)
                predictions = torch.sigmoid(outputs['logits']).cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        y_true = np.array(all_labels)
        y_scores = np.array(all_predictions)
        y_pred = (y_scores > threshold).astype(int)
        
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        per_label_f1 = f1_score(y_true, y_pred, average=None)
        
        results = {
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'per_label_f1': dict(zip(self.label_names, per_label_f1)),
            'threshold': threshold
        }
        
        # Print results
        print(f"\n🎯 DeBERTa Evaluation Results (threshold={threshold}):")
        print(f"  Micro F1: {micro_f1:.3f}")
        print(f"  Macro F1: {macro_f1:.3f}")
        print(f"\n  Per-label F1 scores:")
        for label, score in results['per_label_f1'].items():
            print(f"    {label:<15}: {score:.3f}")
        
        # Detailed classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.label_names,
            zero_division=0
        )
        print(f"\n📋 Detailed Classification Report:")
        print(report)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"results/deberta_evaluation_{timestamp}.json"
        os.makedirs("results", exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'metrics': results,
                'classification_report': report,
                'predictions': y_pred.tolist(),
                'scores': y_scores.tolist(),
                'true_labels': y_true.tolist(),
                'label_names': self.label_names
            }, f, indent=2)
        
        logger.info(f"💾 Evaluation results saved to: {results_file}")
        
        return results

def compare_all_models(deberta_results):
    """Comprehensive comparison of all three models"""
    
    print("\n🏆 Complete Model Comparison")
    print("=" * 60)
    
    # Load all previous results
    models = {}
    
    # DistilBERT
    distilbert_files = [f for f in os.listdir('results') if f.startswith('distilbert_evaluation_')]
    if distilbert_files:
        with open(f"results/{sorted(distilbert_files)[-1]}", 'r') as f:
            models['DistilBERT'] = json.load(f)['metrics']
    
    # RoBERTa
    roberta_files = [f for f in os.listdir('results') if f.startswith('roberta_evaluation_')]
    if roberta_files:
        with open(f"results/{sorted(roberta_files)[-1]}", 'r') as f:
            models['RoBERTa'] = json.load(f)['metrics']
    
    # DeBERTa (current)
    models['DeBERTa'] = deberta_results
    
    # Overall performance comparison
    print(f"📊 Overall Performance:")
    print(f"{'Model':<12} {'Micro F1':<10} {'Macro F1':<10} {'Difference (Micro)':<18}")
    print("-" * 60)
    
    best_micro = max(models.values(), key=lambda x: x['micro_f1'])['micro_f1']
    
    for model_name, results in models.items():
        micro = results['micro_f1']
        macro = results['macro_f1']
        diff = micro - best_micro if micro != best_micro else 0.0
        diff_str = f"{diff:+.3f}" if diff != 0 else "BEST"
        
        print(f"{model_name:<12} {micro:<10.3f} {macro:<10.3f} {diff_str:<18}")
    
    # Per-label comparison
    print(f"\n📊 Per-label F1 Comparison:")
    print(f"{'Label':<15} {'DistilBERT':<12} {'RoBERTa':<10} {'DeBERTa':<10} {'Best':<8}")
    print("-" * 65)
    
    label_winners = {}
    
    for label in deberta_results['per_label_f1']:
        scores = {}
        for model_name, results in models.items():
            if label in results['per_label_f1']:
                scores[model_name] = results['per_label_f1'][label]
        
        best_model = max(scores, key=scores.get)
        best_score = scores[best_model]
        label_winners[label] = (best_model, best_score)
        
        distil_score = scores.get('DistilBERT', 0)
        roberta_score = scores.get('RoBERTa', 0)
        deberta_score = scores.get('DeBERTa', 0)
        
        print(f"{label:<15} {distil_score:<12.3f} {roberta_score:<10.3f} {deberta_score:<10.3f} {best_model:<8}")
    
    # Analysis and recommendations
    print(f"\n🎯 Analysis:")
    
    # Count wins per model
    wins = {}
    total_improvement = {}
    
    for model_name in models.keys():
        wins[model_name] = 0
        total_improvement[model_name] = 0
    
    baseline_model = 'DistilBERT' if 'DistilBERT' in models else list(models.keys())[0]
    
    for label, (winner, score) in label_winners.items():
        wins[winner] += 1
        
        # Calculate improvement over baseline
        if baseline_model in models and label in models[baseline_model]['per_label_f1']:
            baseline_score = models[baseline_model]['per_label_f1'][label]
            for model_name, results in models.items():
                if label in results['per_label_f1']:
                    improvement = results['per_label_f1'][label] - baseline_score
                    total_improvement[model_name] += improvement
    
    print(f"  Category wins: {dict(wins)}")
    print(f"  Best overall micro F1: {max(models.items(), key=lambda x: x[1]['micro_f1'])[0]} ({best_micro:.3f})")
    print(f"  Best overall macro F1: {max(models.items(), key=lambda x: x[1]['macro_f1'])[0]} ({max(models.values(), key=lambda x: x['macro_f1'])['macro_f1']:.3f})")
    
    # Final recommendation
    print(f"\n🚀 Final Recommendation:")
    
    best_model_name = max(models.items(), key=lambda x: x[1]['micro_f1'])[0]
    best_model_score = models[best_model_name]['micro_f1']
    
    print(f"  🏆 Production Model: {best_model_name}")
    print(f"     Micro F1: {best_model_score:.3f}")
    print(f"     Macro F1: {models[best_model_name]['macro_f1']:.3f}")
    
    # Performance tiers
    if best_model_score >= 0.87:
        print(f"     Performance: Excellent (87%+)")
    elif best_model_score >= 0.85:
        print(f"     Performance: Very Good (85-87%)")
    elif best_model_score >= 0.82:
        print(f"     Performance: Good (82-85%)")
    else:
        print(f"     Performance: Baseline (82%)")
    
    # Save comprehensive comparison
    comparison_results = {
        'timestamp': time.strftime("%Y%m%d_%H%M%S"),
        'models': models,
        'label_winners': label_winners,
        'category_wins': wins,
        'best_overall': {
            'model': best_model_name,
            'micro_f1': best_model_score,
            'macro_f1': models[best_model_name]['macro_f1']
        },
        'recommendation': f"Use {best_model_name} for production"
    }
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    comparison_file = f"results/complete_model_comparison_{timestamp}.json"
    
    with open(comparison_file, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n💾 Complete comparison saved to: {comparison_file}")
    
    return comparison_results

def main():
    """Main training and evaluation pipeline"""
    print("🤖 DeBERTa Issue Classifier Training")
    print("=" * 50)
    
    # Check data exists
    data_files = [f for f in os.listdir('data') if f.startswith('sklearn_train_') and f.endswith('.json')]
    if not data_files:
        print("❌ No training data found!")
        print("   Run process_data.py first to prepare the data.")
        return
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize trainer
    trainer = DebertaTrainer()
    
    # Configuration (same as other models)
    config = {
        'epochs': 3,
        'batch_size': 4,
        'learning_rate': 2e-5
    }
    
    print(f"🔧 Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print(f"🔧 Device: {device}")
    
    # Train model
    try:
        train_losses, val_losses, model_path = trainer.train_model(**config)
        print("✅ Training completed successfully!")
        
        # Evaluate model
        results = trainer.evaluate_model()
        
        print(f"\n🎉 Training and evaluation complete!")
        print(f"📊 Final Micro F1: {results['micro_f1']:.3f}")
        print(f"📊 Final Macro F1: {results['macro_f1']:.3f}")
        
        # Complete model comparison
        comparison = compare_all_models(results)
        
        print(f"\n🎯 All encoder models tested!")
        print(f"Ready for LLM comparison phase! 🤖")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()