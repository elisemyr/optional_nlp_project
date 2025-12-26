"""Compare performance of all trained models."""
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_metrics(model_path):
    """Load metrics from a trained model."""
    metrics_file = f"{model_path}/test_metrics.json"
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    return None

def compare_models(output_dir='results/comparison'):
    """Compare all trained models."""
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define models to compare
    models = {
        'XLM-R (es-en)': 'models/xlmr-es-en',
        'mBERT (es-en)': 'models/mbert-es-en',

        'XLM-R (hi-en)': 'models/xlmr-hi-en',
        'mBERT (hi-en)': 'models/mbert-hi-en',

    }
    
    # Collect results
    results = []
    
    for model_name, model_path in models.items():
        metrics = load_metrics(model_path)
        if metrics:
            language = 'Spanish-English' if 'es-en' in model_name else 'Hindi-English'
            model_type = model_name.split()[0]
            
            results.append({
                'Model': model_type,
                'Language': language,
                'F1 (macro)': metrics['test_f1_macro'],
                'Accuracy': metrics['test_accuracy'],
                'F1 (weighted)': metrics.get('test_f1_weighted', metrics['test_f1_macro'])
            })
            print(f"✓ Loaded: {model_name}")
        else:
            print(f"⚠️  Not found: {model_name}")
    
    if not results:
        print("\n❌ No trained models found!")
        print("Train models first using: python src/training/train.py")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(df.to_string(index=False))
    
    # Save to CSV
    csv_path = f'{output_dir}/model_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Saved comparison table to {csv_path}")
    
    # Create visualizations
    
    # 1. F1 Score Comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Spanish-English
    df_es = df[df['Language'] == 'Spanish-English']
    if not df_es.empty:
        ax = axes[0]
        x = np.arange(len(df_es))
        width = 0.35
        
        ax.bar(x - width/2, df_es['F1 (macro)'], width, label='F1 (macro)', alpha=0.8)
        ax.bar(x + width/2, df_es['Accuracy'], width, label='Accuracy', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Spanish-English Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_es['Model'])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, row in enumerate(df_es.itertuples(index=False)):
            ax.text(i - width/2, row[2] + 0.02, f'{row[2]:.3f}', ha='center', fontsize=9)
            ax.text(i + width/2, row[3] + 0.02, f'{row[3]:.3f}', ha='center', fontsize=9)
    
    # Hindi-English
    df_hi = df[df['Language'] == 'Hindi-English']
    if not df_hi.empty:
        ax = axes[1]
        x = np.arange(len(df_hi))
        width = 0.35
        
        ax.bar(x - width/2, df_hi['F1 (macro)'], width, label='F1 (macro)', alpha=0.8)
        ax.bar(x + width/2, df_hi['Accuracy'], width, label='Accuracy', alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Hindi-English Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_hi['Model'])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, row in enumerate(df_hi.itertuples(index=False)):
            ax.text(i - width/2, row[2] + 0.02, f'{row[2]:.3f}', ha='center', fontsize=9)
            ax.text(i + width/2, row[3] + 0.02, f'{row[3]:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plot_path = f'{output_dir}/model_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to {plot_path}")
    plt.close()
    
    # 2. Overall comparison heatmap
    if len(df) >= 3:
        pivot_f1 = df.pivot(index='Model', columns='Language', values='F1 (macro)')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_f1, annot=True, fmt='.4f', cmap='YlGnBu', 
                   cbar_kws={'label': 'F1 Score (macro)'}, ax=ax)
        ax.set_title('Model Performance Comparison (F1 Score)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        heatmap_path = f'{output_dir}/performance_heatmap.png'
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved heatmap to {heatmap_path}")
        plt.close()
    
    # Find best models
    print("\n" + "=" * 70)
    print("BEST MODELS")
    print("=" * 70)
    
    for lang in df['Language'].unique():
        lang_df = df[df['Language'] == lang]
        best_model = lang_df.loc[lang_df['F1 (macro)'].idxmax()]
        print(f"\n{lang}:")
        print(f"  🏆 Best Model: {best_model['Model']}")
        print(f"  📊 F1 Score: {best_model['F1 (macro)']:.4f}")
        print(f"  📊 Accuracy: {best_model['Accuracy']:.4f}")
    
    print("\n" + "=" * 70)
    print("✅ COMPARISON COMPLETE!")
    print("=" * 70)
    
    return df

if __name__ == "__main__":
    compare_models()