"""Error analysis script - analyze model mistakes."""
import argparse
import json
import os
import sys
from pathlib import Path
import random

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.dataset import CodeMixedDataset


def analyze_errors(model_path: str, data_path: str, output_dir: str, num_errors: int = 50):
    """
    Analyze model errors for error analysis section of report.
    
    Args:
        model_path: Path to trained model
        data_path: Path to test data (.pkl)
        output_dir: Directory to save results
        num_errors: Number of errors to analyze per category
    """
    
    print("\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("\n📥 Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Load dataset
    print("📥 Loading dataset...")
    dataset = CodeMixedDataset(data_path, model_path, max_length=128)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Get predictions
    print("🔮 Getting predictions...")
    all_preds = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(-1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
            # Track indices
            batch_start = i * 32
            all_indices.extend(range(batch_start, batch_start + len(labels)))
    
    # Get original data
    df = dataset.df
    
    # Find errors
    errors = []
    for idx, pred, label in zip(all_indices, all_preds, all_labels):
        if pred != label:
            row = df.iloc[idx]
            errors.append({
                'text': row['text'],
                'true_label': label,
                'predicted_label': pred,
                'cs_index': row['cs_index'],
                'words': row['words'],
                'lang_tags': row['lang_tags']
            })
    
    print(f"\n📊 Found {len(errors)} errors out of {len(all_labels)} examples")
    print(f"   Error rate: {len(errors)/len(all_labels)*100:.2f}%")
    
    # Categorize errors
    label_names = {0: 'positive', 1: 'negative', 2: 'neutral'}
    
    error_categories = {
        'pos_to_neg': [],  # Positive misclassified as negative
        'pos_to_neu': [],  # Positive misclassified as neutral
        'neg_to_pos': [],  # Negative misclassified as positive
        'neg_to_neu': [],  # Negative misclassified as neutral
        'neu_to_pos': [],  # Neutral misclassified as positive
        'neu_to_neg': [],  # Neutral misclassified as negative
    }
    
    category_names = {
        'pos_to_neg': 'Positive → Negative',
        'pos_to_neu': 'Positive → Neutral',
        'neg_to_pos': 'Negative → Positive',
        'neg_to_neu': 'Negative → Neutral',
        'neu_to_pos': 'Neutral → Positive',
        'neu_to_neg': 'Neutral → Negative',
    }
    
    for error in errors:
        true_label = label_names[error['true_label']]
        pred_label = label_names[error['predicted_label']]
        
        key = f"{true_label[:3]}_to_{pred_label[:3]}"
        if key in error_categories:
            error_categories[key].append(error)
    
    # Print error distribution
    print("\n📋 Error Distribution:")
    for key, errors_list in error_categories.items():
        if errors_list:
            print(f"   {category_names[key]}: {len(errors_list)} ({len(errors_list)/len(errors)*100:.1f}%)")
    
    # Sample errors for analysis
    print(f"\n📝 Sampling up to {num_errors} errors for detailed analysis...")
    
    sampled_errors = {}
    for key, errors_list in error_categories.items():
        if errors_list:
            sample_size = min(num_errors, len(errors_list))
            sampled_errors[key] = random.sample(errors_list, sample_size)
    
    # Save detailed error analysis
    output_file = f'{output_dir}/error_analysis.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ERROR ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Errors: {len(errors)} / {len(all_labels)} ({len(errors)/len(all_labels)*100:.2f}%)\n\n")
        
        for key, errors_list in sampled_errors.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"{category_names[key].upper()}\n")
            f.write(f"Count: {len(error_categories[key])}\n")
            f.write("=" * 80 + "\n\n")
            
            for i, error in enumerate(errors_list[:10], 1):  # Show first 10
                f.write(f"Example {i}:\n")
                f.write(f"  Text: {error['text']}\n")
                f.write(f"  True: {label_names[error['true_label']]}\n")
                f.write(f"  Predicted: {label_names[error['predicted_label']]}\n")
                f.write(f"  CS-Index: {error['cs_index']:.3f}\n")
                f.write(f"  Words ({len(error['words'])}): {' '.join(error['words'][:15])}{'...' if len(error['words']) > 15 else ''}\n")
                f.write("\n")
    
    print(f"✓ Saved detailed error analysis to {output_file}")
    
    # Save JSON for further analysis
    json_file = f'{output_dir}/errors.json'
    errors_dict = {key: errors_list for key, errors_list in error_categories.items()}
    
    # Convert to serializable format
    serializable_errors = {}
    for key, errors_list in errors_dict.items():
        serializable_errors[key] = [
            {
                'text': e['text'],
                'true_label': label_names[e['true_label']],
                'predicted_label': label_names[e['predicted_label']],
                'cs_index': float(e['cs_index'])
            }
            for e in errors_list
        ]
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_errors, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved errors JSON to {json_file}")
    
    # Analyze error patterns
    print("\n" + "=" * 70)
    print("ERROR PATTERNS")
    print("=" * 70)
    
    # CS-Index analysis
    error_cs_indices = [e['cs_index'] for e in errors]
    avg_error_cs = sum(error_cs_indices) / len(error_cs_indices) if error_cs_indices else 0
    avg_overall_cs = df['cs_index'].mean()
    
    print(f"\nCS-Index Analysis:")
    print(f"  Average CS-Index (errors): {avg_error_cs:.3f}")
    print(f"  Average CS-Index (overall): {avg_overall_cs:.3f}")
    print(f"  Difference: {avg_error_cs - avg_overall_cs:.3f}")
    
    if avg_error_cs > avg_overall_cs:
        print("  → Model struggles more with high CS-density examples")
    else:
        print("  → CS-density not a major factor in errors")
    
    print("\n" + "=" * 70)
    print("✅ ERROR ANALYSIS COMPLETE!")
    print("=" * 70)
    
    return error_categories


def main():
    parser = argparse.ArgumentParser(description='Analyze model errors')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--data', required=True, help='Path to test data (.pkl)')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--num_errors', type=int, default=50, help='Number of errors to sample per category')
    
    args = parser.parse_args()
    
    analyze_errors(args.model, args.data, args.output, args.num_errors)


if __name__ == "__main__":
    main()