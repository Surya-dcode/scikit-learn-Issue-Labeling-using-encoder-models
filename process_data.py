#!/usr/bin/env python3
"""
Data Preprocessing and Analysis for Scikit-learn Issues
Prepares data for LLM and DistilBERT classification comparison
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import re
import os
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IssueDataProcessor:
    """Process and prepare scikit-learn issues for classification"""
    
    def __init__(self, data_file: str):
        """Initialize with path to collected data file"""
        self.data_file = data_file
        self.issues = []
        self.processed_issues = []
        
        # Focus on content-based labels (excluding meta-process labels)
        self.target_labels = {
            'Bug',
            'New Feature', 
            'Documentation',
            'Enhancement',
            'Build / CI',
            'help wanted',  # Keep this as it's content-indicative
            'RFC'  # Request for Comments - technical content
        }
        
    def load_data(self) -> List[Dict]:
        """Load the collected issues data"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.issues = json.load(f)
            logger.info(f"Loaded {len(self.issues)} issues from {self.data_file}")
            return self.issues
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return []
    
    def analyze_original_data(self):
        """Analyze the raw collected data"""
        if not self.issues:
            self.load_data()
            
        print("🔍 Original Data Analysis")
        print("=" * 50)
        print(f"Total issues: {len(self.issues)}")
        
        # Label distribution
        all_labels = []
        for issue in self.issues:
            all_labels.extend(issue.get('labels', []))
        
        label_counts = Counter(all_labels)
        print(f"\nTotal unique labels: {len(label_counts)}")
        print(f"Top 15 labels:")
        for label, count in label_counts.most_common(15):
            pct = (count / len(self.issues)) * 100
            print(f"  {label:<20}: {count:>4} ({pct:>5.1f}%)")
        
        # Text length analysis
        title_lengths = [len(issue['title']) for issue in self.issues]
        body_lengths = [len(issue['body']) for issue in self.issues]
        
        print(f"\nText Statistics:")
        print(f"  Title length - Mean: {np.mean(title_lengths):.1f}, Max: {max(title_lengths)}")
        print(f"  Body length  - Mean: {np.mean(body_lengths):.1f}, Max: {max(body_lengths)}")
        
        return label_counts
    
    def filter_target_issues(self) -> List[Dict]:
        """Filter issues to focus on target labels only"""
        filtered_issues = []
        
        for issue in self.issues:
            issue_labels = set(issue.get('labels', []))
            target_intersection = issue_labels & self.target_labels
            
            if target_intersection:
                # Keep only target labels
                filtered_issue = issue.copy()
                filtered_issue['labels'] = list(target_intersection)
                filtered_issues.append(filtered_issue)
        
        logger.info(f"Filtered to {len(filtered_issues)} issues with target labels")
        return filtered_issues
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
            
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Clean up code blocks and URLs for better readability
        text = re.sub(r'```[\s\S]*?```', '[CODE_BLOCK]', text)
        text = re.sub(r'`[^`\n]+`', '[CODE]', text)
        text = re.sub(r'https?://\S+', '[URL]', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'\[(.*?)\]\([^)]+\)', r'\1', text)  # Links
        
        return text.strip()
    
    def create_combined_text(self, issue: Dict) -> str:
        """Combine title and body into a single text for classification"""
        title = self.clean_text(issue.get('title', ''))
        body = self.clean_text(issue.get('body', ''))
        
        # Combine with clear separator
        combined = f"{title}\n\n{body}"
        
        # Truncate if too long (keep first 1500 chars to stay within token limits)
        if len(combined) > 1500:
            combined = combined[:1500] + "..."
            
        return combined
    
    def process_for_classification(self) -> pd.DataFrame:
        """Process issues into format suitable for classification"""
        filtered_issues = self.filter_target_issues()
        
        processed_data = []
        
        for issue in filtered_issues:
            # Create combined text
            text = self.create_combined_text(issue)
            
            # Skip very short issues
            if len(text.strip()) < 50:
                continue
                
            processed_data.append({
                'number': issue['number'],
                'text': text,
                'labels': issue['labels'],
                'label_count': len(issue['labels']),
                'state': issue['state'],
                'url': issue['url']
            })
        
        df = pd.DataFrame(processed_data)
        logger.info(f"Processed {len(df)} issues for classification")
        
        self.processed_issues = processed_data
        return df
    
    def analyze_processed_data(self, df: pd.DataFrame):
        """Analyze the processed dataset"""
        print("\n📊 Processed Data Analysis")
        print("=" * 50)
        print(f"Final dataset size: {len(df)}")
        
        # Label distribution in final dataset
        all_labels = []
        for labels in df['labels']:
            all_labels.extend(labels)
        
        label_counts = Counter(all_labels)
        print(f"\nFinal label distribution:")
        for label, count in label_counts.most_common():
            pct = (count / len(df)) * 100
            print(f"  {label:<15}: {count:>4} ({pct:>5.1f}%)")
        
        # Multi-label statistics
        label_count_dist = df['label_count'].value_counts().sort_index()
        print(f"\nLabels per issue:")
        for count, freq in label_count_dist.items():
            pct = (freq / len(df)) * 100
            print(f"  {count} label(s): {freq:>4} issues ({pct:>5.1f}%)")
        
        # Text length distribution
        text_lengths = df['text'].str.len()
        print(f"\nText length statistics:")
        print(f"  Mean: {text_lengths.mean():.0f} characters")
        print(f"  Median: {text_lengths.median():.0f} characters")
        print(f"  Max: {text_lengths.max():.0f} characters")
        
        return label_counts
    
    def prepare_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets"""
        # For multi-label, we'll use a simple random split
        # More sophisticated stratification can be added later
        
        train_df, test_df = train_test_split(
            df, 
            test_size=test_size, 
            random_state=42,
            stratify=df['label_count']  # Stratify by number of labels
        )
        
        print(f"\n📋 Data Split:")
        print(f"  Training set: {len(train_df)} issues")
        print(f"  Test set: {len(test_df)} issues")
        
        return train_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save processed data for use in classification"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV for easy inspection
        train_df.to_csv(f'data/sklearn_train_{timestamp}.csv', index=False)
        test_df.to_csv(f'data/sklearn_test_{timestamp}.csv', index=False)
        
        # Save as JSON for programmatic use
        train_df.to_json(f'data/sklearn_train_{timestamp}.json', orient='records', indent=2)
        test_df.to_json(f'data/sklearn_test_{timestamp}.json', orient='records', indent=2)
        
        # Create a simple format for LLM classification
        llm_format = []
        for _, row in train_df.head(100).iterrows():  # Sample for LLM testing
            llm_format.append({
                'text': row['text'],
                'labels': row['labels']
            })
        
        with open(f'data/sklearn_llm_sample_{timestamp}.json', 'w') as f:
            json.dump(llm_format, f, indent=2)
        
        print(f"\n💾 Saved processed data:")
        print(f"  Training: sklearn_train_{timestamp}.csv/.json")
        print(f"  Test: sklearn_test_{timestamp}.csv/.json") 
        print(f"  LLM Sample: sklearn_llm_sample_{timestamp}.json")
        
        return timestamp
    
    def create_visualization(self, df: pd.DataFrame):
        """Create visualizations of the data"""
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Scikit-learn Issues Dataset Analysis', fontsize=16)
        
        # 1. Label distribution
        all_labels = []
        for labels in df['labels']:
            all_labels.extend(labels)
        label_counts = Counter(all_labels)
        
        labels, counts = zip(*label_counts.most_common())
        axes[0, 0].bar(range(len(labels)), counts)
        axes[0, 0].set_xticks(range(len(labels)))
        axes[0, 0].set_xticklabels(labels, rotation=45, ha='right')
        axes[0, 0].set_title('Label Distribution')
        axes[0, 0].set_ylabel('Count')
        
        # 2. Labels per issue
        label_counts_per_issue = df['label_count'].value_counts().sort_index()
        axes[0, 1].bar(label_counts_per_issue.index, label_counts_per_issue.values)
        axes[0, 1].set_title('Number of Labels per Issue')
        axes[0, 1].set_xlabel('Labels per Issue')
        axes[0, 1].set_ylabel('Number of Issues')
        
        # 3. Text length distribution
        text_lengths = df['text'].str.len()
        axes[1, 0].hist(text_lengths, bins=30, alpha=0.7)
        axes[1, 0].set_title('Text Length Distribution')
        axes[1, 0].set_xlabel('Characters')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Issue state distribution
        state_counts = df['state'].value_counts()
        axes[1, 1].pie(state_counts.values, labels=state_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Issue State Distribution')
        
        plt.tight_layout()
        
        # Save plot
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'data/sklearn_analysis_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📈 Visualization saved: sklearn_analysis_{timestamp}.png")

def main():
    """Main processing pipeline"""
    print("🔄 Processing Scikit-learn Issues Data")
    print("=" * 50)
    
    # Find the most recent data file
    data_files = [f for f in os.listdir('data') if f.startswith('sklearn_issues_final_') and f.endswith('.json')]
    if not data_files:
        print("❌ No data files found! Run collect_data.py first.")
        return
    
    latest_file = sorted(data_files)[-1]
    data_path = f"data/{latest_file}"
    print(f"📂 Using data file: {latest_file}")
    
    # Process the data
    processor = IssueDataProcessor(data_path)
    
    # Load and analyze original data
    processor.load_data()
    original_label_counts = processor.analyze_original_data()
    
    # Process for classification
    df = processor.process_for_classification()
    
    if len(df) == 0:
        print("❌ No issues remained after processing!")
        return
    
    # Analyze processed data
    final_label_counts = processor.analyze_processed_data(df)
    
    # Create train/test split
    train_df, test_df = processor.prepare_train_test_split(df)
    
    # Save processed data
    timestamp = processor.save_processed_data(train_df, test_df)
    
    # Create visualizations
    processor.create_visualization(df)
    
    print(f"\n✅ Processing complete!")
    print(f"🎯 Ready for classification experiments!")
    print(f"\nNext steps:")
    print(f"1. LLM Classification: Use sklearn_llm_sample_{timestamp}.json")
    print(f"2. DistilBERT Training: Use sklearn_train_{timestamp}.json")
    print(f"3. Final Evaluation: Use sklearn_test_{timestamp}.json")

if __name__ == "__main__":
    main()