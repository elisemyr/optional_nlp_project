"""Analyze impact of code-switching density on model performance (Goal 5)."""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.dataset import CodeMixedDataset


def analyze_cs_density_impact(model_path: str, data_path: str, output_dir: str):
    """
    Analyze how CS density affects model performance.
    
    This addresses Goal 5: Study impact of code-switching density on model performance
    
    Args:
        model_path: Path to trained model
        data_path: Path to test data (.pkl)
        output_dir: Directory to save results
    """
    
    print("\n" + "=" * 70)
    print("CS DENSITY IMPACT ANALYSIS (Goal 5)")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    print("\n📥 Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Load dataset
    print("📥 Loading dataset...")
    dataset = CodeMixedDataset(data_path, model_path, max_length=128)
    
    # Get CS-Index for all examples
    df = dataset.df
    print(f"✓ Loaded {len(df)} examples")
    print(f"  CS-Index range: [{df['cs_index'].min():.3f}, {df['cs_index'].max():.3f}]")
    print(f"  CS-Index mean: {df['cs_index'].mean():.3f} ± {df['cs_index'].std():.3f}")
    
    # Define CS density bins
    bins = [
        (0.0, 0.3, 'Low (0-0.3)'),
        (0.3, 0.6, 'Medium (0.3-0.6)'),
        (0.6, 1.0, 'High (0.6+)')
    ]
    
    results = []
    
    print("\n🔍 Analyzing performance by CS density bin...")
    
    for min_cs, max_cs, label in bins:
        # Filter examples in this bin
        if max_cs == 1.0:
            mask = (df['cs_index'] >= min_cs) & (df['cs_index'] <= max_cs)
        else:
            mask = (df['cs_index'] >= min_cs) & (df['cs_index'] < max_cs)
        
        bin_indices = df[mask].index.tolist()
        
        if len(bin_indices) == 0:
            print(f"\n⚠️  {label}: No examples found")
            continue
        
        print(f"\n📊 {label}:")
        print(f"   Examples: {len(bin_indices)}")
        
        # Create subset
        subset = Subset(dataset, bin_indices)
        dataloader = DataLoader(subset, batch_size=32, shuffle=False)
        
        # Get predictions
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"  {label}", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels']
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(-1).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        f1 = f1_score(all_labels, all_preds, average='macro')
        acc = accuracy_score(all_labels, all_preds)
        
        print(f"   F1 Score: {f1:.4f}")
        print(f"   Accuracy: {acc:.4f}")
        
        # Calculate per-class F1
        f1_per_class = f1_score(all_labels, all_preds, average=None)
        
        results.append({
            'bin': label,
            'min_cs': min_cs,
            'max_cs': max_cs,
            'count': len(bin_indices),
            'f1_macro': float(f1),
            'accuracy': float(acc),
            'f1_positive': float(f1_per_class[0]),
            'f1_negative': float(f1_per_class[1]),
            'f1_neutral': float(f1_per_class[2])
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Plot F1 vs CS Density
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: F1 Score by CS Density
    x = range(len(results_df))
    ax1.bar(x, results_df['f1_macro'], color='steelblue', alpha=0.7)
    ax1.set_xlabel('CS Density Bin', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1 Score (macro)', fontsize=12, fontweight='bold')
    ax1.set_title('Model Performance vs Code-Switching Density', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['bin'])
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(results_df.iterrows()):
        ax1.text(i, row['f1_macro'] + 0.02, f"{row['f1_macro']:.3f}", 
                ha='center', fontweight='bold')
        ax1.text(i, 0.05, f"n={row['count']}", ha='center', fontsize=9)
    
    # Plot 2: Per-class F1 by CS Density
    width = 0.25
    x_pos = np.arange(len(results_df))
    
    ax2.bar(x_pos - width, results_df['f1_positive'], width, label='Positive', alpha=0.8)
    ax2.bar(x_pos, results_df['f1_negative'], width, label='Negative', alpha=0.8)
    ax2.bar(x_pos + width, results_df['f1_neutral'], width, label='Neutral', alpha=0.8)
    
    ax2.set_xlabel('CS Density Bin', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    ax2.set_title('Per-Class Performance vs CS Density', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(results_df['bin'])
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plot_path = f'{output_dir}/cs_density_impact.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to {plot_path}")
    plt.close()
    
    # Calculate performance degradation
    degradation = 0.0
    if len(results_df) >= 2:
        low_f1 = results_df.iloc[0]['f1_macro']
        high_f1 = results_df.iloc[-1]['f1_macro']
        degradation = (low_f1 - high_f1) / low_f1 * 100
        
        print(f"\n📉 Performance Analysis:")
        print(f"   Low CS F1: {low_f1:.4f}")
        print(f"   High CS F1: {high_f1:.4f}")
        print(f"   Degradation: {degradation:.1f}%")
    
    # Save results
    results_path = f'{output_dir}/cs_density_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to {results_path}")
    
    # Save CSV
    csv_path = f'{output_dir}/cs_density_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV to {csv_path}")
    
    print("\n" + "=" * 70)
    print("✅ CS DENSITY ANALYSIS COMPLETE!")
    print("=" * 70)
    
    if len(results_df) >= 2:
        print("\nKey Finding:")
        print(f"Model performance {'decreases' if degradation > 0 else 'increases'} by {abs(degradation):.1f}% ")
        print(f"from low to high code-switching density.")
    else:
        print("\n⚠️  Insufficient CS density variation for comparison.")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Analyze CS density impact on performance')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--data', required=True, help='Path to test data (.pkl)')
    parser.add_argument('--output', required=True, help='Output directory')
    
    args = parser.parse_args()
    
    analyze_cs_density_impact(args.model, args.data, args.output)


if __name__ == "__main__":
    main()