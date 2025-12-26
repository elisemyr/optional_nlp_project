"""Evaluation script for trained sentiment classification models."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.dataset import CodeMixedDataset


def evaluate_model(model_path: str, data_path: str, output_dir: str):
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to trained model directory
        data_path: Path to test data (.pkl file)
        output_dir: Directory to save evaluation results
    """
    
    print("\n" + "=" * 70)
    print(f"EVALUATING MODEL")
    print("=" * 70)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("\n📥 Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    print(f"✓ Model loaded on {device}")
    
    # Load test data
    print("\n📥 Loading test data...")
    dataset = CodeMixedDataset(data_path, model_path, max_length=128)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    print(f"✓ Loaded {len(dataset)} examples")
    
    # Get predictions
    print("\n🔮 Running inference...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # Get predictions
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    print("\n" + "=" * 70)
    print("📊 RESULTS")
    print("=" * 70)
    
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    print(f"\n✓ Accuracy: {accuracy:.4f}")
    print(f"✓ F1 Score (macro): {f1_macro:.4f}")
    print(f"✓ F1 Score (weighted): {f1_weighted:.4f}")
    print(f"✓ Precision (macro): {precision:.4f}")
    print(f"✓ Recall (macro): {recall:.4f}")
    
    # Classification report
    label_names = ['Positive', 'Negative', 'Neutral']
    report = classification_report(
        all_labels, all_preds,
        target_names=label_names,
        digits=4
    )
    print("\n📋 Classification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    cm_path = f'{output_dir}/confusion_matrix.png'
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved confusion matrix to {cm_path}")
    plt.close()
    
    # Per-class metrics
    per_class_f1 = f1_score(all_labels, all_preds, average=None)
    per_class_precision = precision_score(all_labels, all_preds, average=None)
    per_class_recall = recall_score(all_labels, all_preds, average=None)
    
    # Plot per-class metrics
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(label_names))
    width = 0.25
    
    ax.bar(x - width, per_class_precision, width, label='Precision', alpha=0.8)
    ax.bar(x, per_class_recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, per_class_f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(label_names)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    metrics_plot_path = f'{output_dir}/per_class_metrics.png'
    plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-class metrics to {metrics_plot_path}")
    plt.close()
    
    # Save metrics to JSON
    metrics = {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'per_class_metrics': {
            'positive': {
                'precision': float(per_class_precision[0]),
                'recall': float(per_class_recall[0]),
                'f1': float(per_class_f1[0])
            },
            'negative': {
                'precision': float(per_class_precision[1]),
                'recall': float(per_class_recall[1]),
                'f1': float(per_class_f1[1])
            },
            'neutral': {
                'precision': float(per_class_precision[2]),
                'recall': float(per_class_recall[2]),
                'f1': float(per_class_f1[2])
            }
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    metrics_path = f'{output_dir}/metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics to {metrics_path}")
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds,
        'correct': all_labels == all_preds,
        'prob_positive': all_probs[:, 0],
        'prob_negative': all_probs[:, 1],
        'prob_neutral': all_probs[:, 2]
    })
    
    predictions_path = f'{output_dir}/predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    print(f"✓ Saved predictions to {predictions_path}")
    
    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE!")
    print("=" * 70)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--model', required=True, help='Path to trained model directory')
    parser.add_argument('--data', required=True, help='Path to test data (.pkl file)')
    parser.add_argument('--output', required=True, help='Output directory for results')
    
    args = parser.parse_args()
    
    evaluate_model(args.model, args.data, args.output)


if __name__ == "__main__":
    main()