#!/usr/bin/env python3
"""
DistilBERT Multi-label Issue Classifier for Scikit-learn Issues
Optimized for MacBook Pro M4 with Apple Silicon
"""

import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import classification_report, multilabel_confusion_matrix, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import List, Dict, Tuple
import time
from tqdm.auto import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for Apple Silicon optimization
if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("🍎 Using Apple Silicon MPS acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    logger.info("🚀 Using CUDA GPU")
else:
    device = torch.device("cpu")
    logger.info("💻 Using CPU")

class IssueDataset(Dataset):
    """Dataset class for scikit-learn issues"""
    
    def __init__(self, texts: List[str], labels: List[List[str]], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Create label binarizer
        self.label_names = ['Bug', 'New Feature', 'Documentation', 'Build / CI', 
                           'help wanted', 'RFC', 'Enhancement']
        self.mlb = MultiLabelBinarizer(classes=self.label_names)
        self.binary_labels = self.mlb.fit_transform(labels)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        labels = self.binary_labels[idx].astype(np.float32)
        
        # Tokenize
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

class DistilBertMultiLabelClassifier(nn.Module):
    """Custom DistilBERT model for multi-label classification"""
    
    def __init__(self, num_labels: int, dropout_rate: float = 0.3):
        super().__init__()
        self.num_labels = num_labels
        
        # Load pre-trained DistilBERT
        self.distilbert = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Classification head
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get DistilBERT outputs
        outputs = self.distilbert.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0]
        pooled_output = self.dropout(pooled_output)
        
        # Get logits
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            # Binary cross entropy loss for multi-label
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        
        return {
            'loss': loss,
            'logits': logits
        }

class IssueClassifierTrainer:
    """Trainer class for the issue classifier"""
    
    def __init__(self, model_name: str = 'distilbert-base-uncased'):
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.label_names = ['Bug', 'New Feature', 'Documentation', 'Build / CI', 
                           'help wanted', 'RFC', 'Enhancement']
        self.num_labels = len(self.label_names)
        
        # Model will be initialized during training
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data"""
        # Find the latest processed data files
        data_files = [f for f in os.listdir('data') if f.startswith('sklearn_train_') and f.endswith('.json')]
        if not data_files:
            raise FileNotFoundError("No training data found! Run process_data.py first.")
        
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
        # Create datasets
        self.train_dataset = IssueDataset(
            texts=train_df['text'].tolist(),
            labels=train_df['labels'].tolist(),
            tokenizer=self.tokenizer,
            max_length=max_length
        )
        
        self.val_dataset = IssueDataset(
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
    
    def create_data_loaders(self, batch_size: int = 8) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders with appropriate batch size for Mac"""
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # Important for Mac
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=0
        )
        
        return train_loader, val_loader
    
    def train_model(self, 
                   epochs: int = 3,
                   batch_size: int = 8,
                   learning_rate: float = 2e-5,
                   warmup_steps: int = 100,
                   max_length: int = 512):
        """Train the DistilBERT model"""
        
        logger.info("🚀 Starting DistilBERT training...")
        
        # Load data
        train_df, test_df = self.load_data()
        
        # Prepare datasets
        self.prepare_datasets(train_df, test_df, max_length)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(batch_size)
        
        # Initialize model
        self.model = DistilBertMultiLabelClassifier(self.num_labels)
        self.model.to(device)
        
        # Optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
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
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                
                # Backward pass
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
        model_path = f"models/distilbert_classifier_{timestamp}"
        os.makedirs("models", exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'label_names': self.label_names,
            'tokenizer': self.tokenizer,
            'training_args': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'max_length': max_length
            }
        }, f"{model_path}.pth")
        
        logger.info(f"💾 Model saved to: {model_path}.pth")
        
        # Plot training curves
        self.plot_training_curves(train_losses, val_losses, timestamp)
        
        return train_losses, val_losses
    
    def plot_training_curves(self, train_losses: List[float], val_losses: List[float], timestamp: str):
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.title('DistilBERT Training Progress', fontsize=16)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/training_curves_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"📈 Training curves saved to: results/training_curves_{timestamp}.png")
    
    def evaluate_model(self, threshold: float = 0.5) -> Dict:
        """Evaluate the trained model"""
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
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_scores = np.array(all_predictions)
        y_pred = (y_scores > threshold).astype(int)
        
        # Calculate metrics
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Per-label metrics
        per_label_f1 = f1_score(y_true, y_pred, average=None)
        
        results = {
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'per_label_f1': dict(zip(self.label_names, per_label_f1)),
            'threshold': threshold
        }
        
        # Print results
        print(f"\n🎯 Evaluation Results (threshold={threshold}):")
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
        results_file = f"results/distilbert_evaluation_{timestamp}.json"
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

def main():
    """Main training and evaluation pipeline"""
    print("🤖 DistilBERT Issue Classifier Training")
    print("=" * 50)
    
    # Check if data exists
    data_files = [f for f in os.listdir('data') if f.startswith('sklearn_train_') and f.endswith('.json')]
    if not data_files:
        print("❌ No training data found!")
        print("   Run process_data.py first to prepare the data.")
        return
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize trainer
    trainer = IssueClassifierTrainer()
    
    # Configuration
    config = {
        'epochs': 3,
        'batch_size': 4,  # Small batch size for Mac
        'learning_rate': 2e-5,
        'max_length': 512
    }
    
    print(f"🔧 Training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print(f"🔧 Device: {device}")
    
    # Train model
    try:
        train_losses, val_losses = trainer.train_model(**config)
        print("✅ Training completed successfully!")
        
        # Evaluate model
        results = trainer.evaluate_model()
        
        print(f"\n🎉 Training and evaluation complete!")
        print(f"📊 Final Micro F1: {results['micro_f1']:.3f}")
        print(f"📊 Final Macro F1: {results['macro_f1']:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()